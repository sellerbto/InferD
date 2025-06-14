from typing import List, Optional
import uuid
import torch
from torch import nn
from transformers import AutoTokenizer, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from models.qwen3.qwen3_config import Qwen3Config
from chain_rpc_client import ChainRPCQwen3Client

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.rope_type = "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(
            device=device, base=Qwen3Config.ROPE_THETA, dim=Qwen3Config.HEAD_DIM
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv.float() @ pos.float()).transpose(1,2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling
class Qwen3Client(nn.Module):
    def __init__(self, server_addrs: List[tuple]):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.embed_tokens = nn.Embedding(
            Qwen3Config.VOCAB_SIZE, Qwen3Config.HIDDEN_SIZE, Qwen3Config.PAD_TOKEN_ID
        )
        self.norm = Qwen3RMSNorm(Qwen3Config.HIDDEN_SIZE, Qwen3Config.RMS_NORM_EPS)
        self.lm_head = nn.Linear(Qwen3Config.HIDDEN_SIZE, Qwen3Config.VOCAB_SIZE, bias=False)
        self.rotary_emb = Qwen3RotaryEmbedding()
        self.to(self.device)
        self._load_weights()
        self.logit_processors = LogitsProcessorList([
            TemperatureLogitsWarper(Qwen3Config.TEMPERATURE),
            TopKLogitsWarper(Qwen3Config.TOP_K),
            TopPLogitsWarper(Qwen3Config.TOP_P),
        ])
        self._dummy_ids = torch.zeros((1,1), dtype=torch.long, device=self.device)
        self.rpc = ChainRPCQwen3Client(server_addrs)

    def _choose_next(self, logits: torch.Tensor) -> torch.Tensor:
        proc = self.logit_processors(self._dummy_ids, logits)
        return torch.multinomial(torch.softmax(proc, dim=-1), num_samples=1)

    def _should_continue(self, ids: List[int], max_length: Optional[int]) -> bool:
        if max_length and len(ids) >= max_length: return False
        if ids and ids[-1] == Qwen3Config.EOS_TOKEN_ID: return False
        return True

    def _load_weights(self):
        for fname, target_module in [
            ("embed_tokens.pt", self.embed_tokens),
            ("norm.pt", self.norm),
            ("lm_head.pt", self.lm_head),
        ]:
            file_path = f'/models/inferd-qwen3/{fname}'
            state_dict = torch.load(file_path, map_location=self.device)
            target_module.load_state_dict(state_dict)

    async def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        sess = str(uuid.uuid4())
        print(f"Session {sess}, prompt: {prompt}")
        with torch.no_grad():
            ids: List[int] = []
            # prepare prompt
            text = self.tokenizer.apply_chat_template(
                [{"role":"user","content":prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            emb = self.embed_tokens(input_ids)
            B,S,_ = emb.shape
            pos_ids = torch.arange(S, device=self.device).unsqueeze(0)
            dtype = emb.dtype
            tril = torch.tril(torch.ones(S,S,device=self.device,dtype=dtype))
            mask = ((1-tril)*torch.finfo(dtype).min).unsqueeze(0).unsqueeze(0).expand(B,1,S,S)
            cos,sin = self.rotary_emb(emb, pos_ids)
            # initial forward
            hs = await self.rpc.forward_through_chain(
                hidden_states=emb, attention_mask=mask,
                cache_position=pos_ids.squeeze(0), cos=cos, sin=sin,
                session_id=sess, timeout=60.0, start_stage=0, use_cache=False
            )
            # first token
            out = self.lm_head(self.norm(hs[:,-1]))
            tok = self._choose_next(out)
            ids.append(int(tok.item()))
            # incremental loop
            while self._should_continue(ids, max_length):
                emb1 = self.embed_tokens(tok.unsqueeze(0))
                t = input_ids.shape[1] + len(ids)-1
                pos = torch.tensor([[t]],device=self.device)
                cos1,sin1 = self.rotary_emb(emb1,pos)
                hs = await self.rpc.forward_through_chain(
                    hidden_states=emb1, attention_mask=torch.zeros((1,1,1,1), device=self.device, dtype=dtype),
                    cache_position=pos.squeeze(0), cos=cos1, sin=sin1,
                    session_id=sess, timeout=60.0, start_stage=0, use_cache=True
                )
                out = self.lm_head(self.norm(hs[:,-1]))
                tok = self._choose_next(out)
                ids.append(int(tok.item()))
                print(f'DECODED = {self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)}')
            return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
