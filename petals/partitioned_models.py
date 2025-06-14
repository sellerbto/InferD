import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer
import base64
import numpy as np
from typing import List
from torch import nn
import yaml
from transformers import (
    AutoTokenizer,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def tensor_to_base64(tensor: torch.Tensor) -> dict:
    array = tensor.detach().cpu().numpy()
    b64 = base64.b64encode(array.tobytes()).decode("utf-8")
    return {
        "b64": b64,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }

def base64_to_tensor(meta: dict) -> torch.Tensor:
    b64 = meta["b64"]
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    data = base64.b64decode(b64)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return torch.from_numpy(array)

def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
    seq_len = attn_mask_2d.size(1)
    causal = torch.tril(torch.ones((seq_len, seq_len),
                                   device=attn_mask_2d.device,
                                   dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
    padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
    return padding_mask & causal  # (1,1,seq_len,seq_len)


# === Определяем стадии ===

class FirstStage(nn.Module):
    def __init__(self, embed_tokens, rotary_emb, layers: List[nn.Module]):
        super().__init__()
        self.embed  = embed_tokens
        self.rotary = rotary_emb
        self.layers = nn.ModuleList(layers)

    def forward(self, input_ids, decoder_attn_mask, position_ids):
        hidden_states = self.embed(input_ids)
        cos, sin = self.rotary(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=decoder_attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin)
            )[0]
        return hidden_states


class StageInner(nn.Module):
    def __init__(self, rotary_emb, layers: List[nn.Module]):
        super().__init__()
        self.rotary = rotary_emb
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states, decoder_attn_mask, position_ids):
        cos, sin = self.rotary(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=decoder_attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin)
            )[0]
        return hidden_states


class LastStage(nn.Module):
    def __init__(self, rotary_emb, layers: List[nn.Module], final_norm, lm_head):
        super().__init__()
        self.rotary  = rotary_emb
        self.layers  = nn.ModuleList(layers)
        self.norm    = final_norm
        self.lm_head = lm_head

    def forward(self, hidden_states, decoder_attn_mask, position_ids):
        cos, sin = self.rotary(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=decoder_attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin)
            )[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

from torch.serialization import add_safe_globals
add_safe_globals([FirstStage, StageInner, LastStage])

class PartitionedQwen2:
    def __init__(self, model_name: str, num_stages: int, stage: int, parts_path: str):
        self.stage  = stage
        self.num_stages = num_stages
        self.parts_path = parts_path
        print(f"PartitionedQwen2: stage {self.stage}, num_stages = {self.num_stages}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if stage == 0 or stage == self.num_stages - 1:
            self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        self.model = torch.load(
            self.parts_path,
            map_location=self.device,
            weights_only=False
        )
        self.model.to(self.device).eval()

    def _prepare_inputs(self, input_data):
        if self.stage == 0:
            if isinstance(input_data, str):
                enc = self.tokenizer(input_data, return_tensors="pt")
                ids = enc.input_ids.to(self.device)
                return ids.squeeze().tolist(), ids
            if "prompt" in input_data:
                enc = self.tokenizer(input_data["prompt"], return_tensors="pt")
                ids = enc.input_ids.to(self.device)
                return ids.squeeze().tolist(), ids
            if "generated_ids" in input_data:
                lst = input_data["generated_ids"]
                return lst, torch.tensor([lst], device=self.device, dtype=torch.long)

        if "hidden_meta" in input_data:
            hidden = base64_to_tensor(input_data["hidden_meta"]).to(self.device)
            return input_data.get("generated_ids"), hidden

        raise RuntimeError(f"Bad input for stage {self.stage}: {input_data!r}")

    def _create_attention_mask(self, seq_len: int):
        att1 = torch.ones((1, seq_len), device=self.device, dtype=torch.long)
        mask = build_decoder_attention_mask(att1)
        pos  = torch.arange(seq_len, device=self.device).unsqueeze(0)
        return mask, pos

    def forward(self, inputs: dict) -> dict:
        gen_ids, model_in = self._prepare_inputs(inputs)
        seq_len = model_in.size(1) if isinstance(model_in, torch.Tensor) else len(model_in)
        attn_mask, pos_ids = self._create_attention_mask(seq_len)

        with torch.no_grad():
            out = self.model(model_in, attn_mask, pos_ids)

        if self.stage < self.num_stages - 1:
            if self.stage == 0:
                return {
                    "hidden_meta":   tensor_to_base64(out),
                    "generated_ids": gen_ids
                }
            return {"hidden_meta": tensor_to_base64(out)}

        logits = out[:, -1, :]
        token_id_tensor = self._choose_next(logits)  # shape [1,1]
        token_id = token_id_tensor.item()
        token_str = self.tokenizer.decode(token_id)

        return {
            "next_token_id":  token_id,
            "next_token_str": token_str,
            "generated_ids":  gen_ids + [token_id]
        }

    def _choose_next(self, logits: torch.Tensor) -> torch.Tensor:
        logit_processors = LogitsProcessorList(
            [
                TemperatureLogitsWarper(0.6),
                TopKLogitsWarper(20),
                TopPLogitsWarper(0.95),
            ]
        )
        _dummy_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        logits = logits.clone()
        logits = logit_processors(_dummy_input_ids, logits)
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        return next_token_id
