from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN
from transformers import DynamicCache
from torch import nn
import torch
from huggingface_hub import hf_hub_download

from qwen3_config import Qwen3Config

from collections import defaultdict


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


class Qwen3MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = Qwen3Config.HIDDEN_SIZE
        self.intermediate_size = Qwen3Config.INTERMEDIATE_SIZE
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[Qwen3Config.HIDDEN_ACT]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = Qwen3Config.HEAD_DIM
        self.num_key_value_groups = (
            Qwen3Config.NUM_ATTENTION_HEADS // Qwen3Config.NUM_KEY_VALUE_HEADS
        )
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = nn.Linear(
            Qwen3Config.HIDDEN_SIZE,
            Qwen3Config.NUM_ATTENTION_HEADS * self.head_dim,
            bias=Qwen3Config.ATTENTION_BIAS,
        )
        self.k_proj = nn.Linear(
            Qwen3Config.HIDDEN_SIZE,
            Qwen3Config.NUM_KEY_VALUE_HEADS * self.head_dim,
            bias=Qwen3Config.ATTENTION_BIAS,
        )
        self.v_proj = nn.Linear(
            Qwen3Config.HIDDEN_SIZE,
            Qwen3Config.NUM_KEY_VALUE_HEADS * self.head_dim,
            bias=Qwen3Config.ATTENTION_BIAS,
        )
        self.o_proj = nn.Linear(
            Qwen3Config.NUM_ATTENTION_HEADS * self.head_dim,
            Qwen3Config.HIDDEN_SIZE,
            bias=Qwen3Config.ATTENTION_BIAS,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=Qwen3Config.RMS_NORM_EPS)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=Qwen3Config.RMS_NORM_EPS)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.hidden_size = Qwen3Config.HIDDEN_SIZE

        self.self_attn = Qwen3Attention(layer_idx)

        self.mlp = Qwen3MLP()
        self.input_layernorm = Qwen3RMSNorm(Qwen3Config.HIDDEN_SIZE, eps=Qwen3Config.RMS_NORM_EPS)
        self.post_attention_layernorm = Qwen3RMSNorm(
            Qwen3Config.HIDDEN_SIZE, eps=Qwen3Config.RMS_NORM_EPS
        )
        self.attention_type = Qwen3Config.ATTENTION_TYPE

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Server(nn.Module):
    def __init__(self, start_layer: int, end_layer: int):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.session_caches = defaultdict(DynamicCache)
        self.local_layers = nn.ModuleList(
            [Qwen3DecoderLayer(layer_idx) for layer_idx in range(start_layer, end_layer + 1)]
        )
        self.to(self.device)
        self._load_weights()

    def _load_weights(self):
        for layer in self.local_layers:
            idx = layer.self_attn.layer_idx
            fname = f"layer_{idx:02d}.pt"

            cached_path = hf_hub_download(repo_id=Qwen3Config.HF_REPO_ID, filename=fname)

            layer_sd = torch.load(cached_path, map_location=self.device)
            layer.load_state_dict(layer_sd)

    def send(
        self,
        session_id,
        hidden_states,
        attention_mask,
        cache_position,
        position_embeddings,
        input_start_layer=None,
        input_end_layer=None,
    ):
        for layer in self.local_layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                past_key_value=self.session_caches[session_id]
            )
        return hidden_states
