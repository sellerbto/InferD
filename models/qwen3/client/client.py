from collections import deque
from typing import List, Optional
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers import (
    AutoTokenizer,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from torch import nn
import torch
from huggingface_hub import hf_hub_download

import uuid

from qwen3_config import Qwen3Config
import io
import torch
import grpc
from proto import qwen3_pb2, qwen3_pb2_grpc
from client.rpc_client import RPCQwen3Client


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
        self.max_seq_len_cached = Qwen3Config.MAX_POSITION_EMBEDDINGS
        self.original_max_seq_len = Qwen3Config.MAX_POSITION_EMBEDDINGS

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            device=device, base=Qwen3Config.ROPE_THETA, dim=Qwen3Config.HEAD_DIM
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Client(nn.Module):
    def __init__(self, server_addrs: List[tuple]):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        self.embed_tokens = nn.Embedding(
            Qwen3Config.VOCAB_SIZE, Qwen3Config.HIDDEN_SIZE, Qwen3Config.PAD_TOKEN_ID
        )
        self.norm = Qwen3RMSNorm(Qwen3Config.HIDDEN_SIZE, Qwen3Config.RMS_NORM_EPS)
        self.lm_head = nn.Linear(Qwen3Config.HIDDEN_SIZE, Qwen3Config.VOCAB_SIZE, bias=False)
        self.rotary_emb = Qwen3RotaryEmbedding()
        self.to(self.device)
        self._load_weights()

        self.logit_processors = LogitsProcessorList(
            [
                TemperatureLogitsWarper(Qwen3Config.TEMPERATURE),
                TopKLogitsWarper(Qwen3Config.TOP_K),
                TopPLogitsWarper(Qwen3Config.TOP_P),
            ]
        )
        self._dummy_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.rpc_client = RPCQwen3Client(server_addrs)

    def _load_weights(self):
        for fname, target_module in [
            ("embed_tokens.pt", self.embed_tokens),
            ("norm.pt", self.norm),
            ("lm_head.pt", self.lm_head),
        ]:
            file_path = hf_hub_download(repo_id=Qwen3Config.HF_REPO_ID, filename=fname)
            state_dict = torch.load(file_path, map_location=self.device)
            target_module.load_state_dict(state_dict)

    def _choose_next(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        logits = self.logit_processors(self._dummy_input_ids, logits)
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        return next_token_id

    def _should_continue(self, generated_ids: List[int], max_length: Optional[int]) -> bool:
        if max_length is not None and len(generated_ids) >= max_length:
            return False

        if len(generated_ids) > 0 and generated_ids[-1] == Qwen3Config.EOS_TOKEN_ID:
            return False

        return True

    def _find_best_chain(self, known_servers):
        return deque(
            [
                Qwen3Server(0, 9),
                Qwen3Server(10, 19),
                Qwen3Server(20, Qwen3Config.NUM_HIDDEN_LAYERS - 1),
            ]
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=cache_position.device,
            )
            diagonal_attend_mask = torch.arange(
                target_length, device=cache_position.device
            ) > cache_position.reshape(-1, 1)

            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

    def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        session_id = str(uuid.uuid4())
        with torch.no_grad():
            generated_ids: List[int] = []
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            all_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

            input_embeds = self.embed_tokens(all_input_ids)
            batch_size, seq_len = input_embeds.shape[0], input_embeds.shape[1]
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            dtype = input_embeds.dtype
            min_val = torch.finfo(dtype).min

            tril_2d = torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=dtype))
            causal_mask_4d = ((1.0 - tril_2d) * min_val).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

            cos, sin = self.rotary_emb(input_embeds, position_ids)
            hidden_states = input_embeds

            hidden_states = self.rpc_client.forward_through_chain(
                hidden_states=hidden_states,
                attention_mask=causal_mask_4d,
                cache_position=position_ids.squeeze(0),
                cos=cos,
                sin=sin,
                session_id=session_id,
            )

            hidden_last = hidden_states[:, -1, :]
            hidden_norm = self.norm(hidden_last)
            last_logits = self.lm_head(hidden_norm)
            next_token_id = self._choose_next(last_logits)  # [1,1]
            generated_ids.append(next_token_id.item())

            while self._should_continue(generated_ids, max_length):
                if isinstance(next_token_id, torch.Tensor):
                    single_input_ids = next_token_id
                else:
                    single_input_ids = torch.tensor([[next_token_id]], device=self.device)
                emb = self.embed_tokens(single_input_ids)  # [1,1,hidden_size]

                small_mask2d = torch.zeros((1, 1), device=self.device, dtype=dtype)
                causal_mask_single = small_mask2d.unsqueeze(0).unsqueeze(0)  # [1,1,1,1]

                past_len = all_input_ids.shape[1] + len(generated_ids) - 1
                position_ids = torch.tensor([[past_len]], device=self.device)  # [1,1]

                cos, sin = self.rotary_emb(emb, position_ids)

                hidden_states = self.rpc_client.forward_through_chain(
                    hidden_states=emb,
                    attention_mask=causal_mask_single,
                    cache_position=position_ids.squeeze(0),
                    cos=cos,
                    sin=sin,
                    session_id=session_id,
                )

                hidden_last = hidden_states[:, -1, :]
                hidden_norm = self.norm(hidden_last)
                last_logits = self.lm_head(hidden_norm)
                next_token_id = self._choose_next(last_logits)
                generated_ids.append(next_token_id.item())

                # print(
                #     self.tokenizer.decode(
                #         [next_token_id.item()],
                #         skip_special_tokens=True,
                #         clean_up_tokenization_spaces=True,
                #     ),
                #     end="",
                #     flush=True,
                # )

            output_str = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return output_str
        


 