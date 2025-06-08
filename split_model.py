import yaml
import torch
from transformers import Qwen2ForCausalLM
import os
from typing import List
from torch import nn
import torch
import torch.nn.functional as F
from typing import List
from torch import nn


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


with open("petals/inferd.yaml") as f:
    cfg = yaml.safe_load(f)
parts_dir = cfg["parts_dir"]
stages   = cfg["stages"]

full = Qwen2ForCausalLM.from_pretrained(cfg["model_name"])
embed_tokens = full.model.embed_tokens
rotary_emb    = full.model.rotary_emb
all_layers    = full.model.layers
final_norm    = full.model.norm
lm_head       = full.lm_head


os.mkdir(parts_dir)
print(f"✔ Loaded full model: layers = {len(all_layers)}")
num_stages = len(stages)
for idx, st in enumerate(stages):
    name    = st["name"]
    sl, el  = st["start_layer"], st["end_layer"]
    sublayers = all_layers[sl:el+1]

    if idx == 0:
        module = FirstStage(embed_tokens, rotary_emb, sublayers)
    elif idx == num_stages - 1:
        module = LastStage(rotary_emb, sublayers, final_norm, lm_head)
    else:
        module = StageInner(rotary_emb, sublayers)

    module.eval()
    os.mkdir(f"{parts_dir}/{name}")
    path_to_save = f"{parts_dir}/{name}/model.pth"
    torch.save(module, path_to_save)
    print(f"Saved {path_to_save}")
os.mkdir(f"{parts_dir}/test")
