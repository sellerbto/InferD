import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer
import base64
import numpy as np
# === Вспомогательные функции для конвертации Tensor ↔ base64 ===

def tensor_to_base64(tensor: torch.Tensor) -> dict:
    """
    Возвращает:
      {
        "b64": "<сами байты в base64>",
        "dtype": "float32",
        "shape": [1, seq_len, hidden_size]
      }
    """
    array = tensor.detach().cpu().numpy()
    b64 = base64.b64encode(array.tobytes()).decode("utf-8")
    return {
        "b64":   b64,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }

def base64_to_tensor(meta: dict) -> torch.Tensor:
    """
    Ожидает dict из tensor_to_base64:
      {
        "b64": "<строка>",
        "dtype": "float32",
        "shape": [1, seq_len, hidden_size]
      }
    Возвращает torch.Tensor той же формы.
    """
    b64 = meta["b64"]
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    data = base64.b64decode(b64)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return torch.from_numpy(array)

# === Функция построения 4D каузальной маски из 2D padding-маски ===

def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
    """
    Формирует 4D-каузальную маску из 2D padding-маски (все единицы, если нет паддинга).
    attn_mask_2d: (1, seq_len), LongTensor из единиц.
    Возвращает BoolTensor (1, 1, seq_len, seq_len).
    """
    seq_len = attn_mask_2d.size(1)
    causal = torch.tril(torch.ones((seq_len, seq_len),
                                   device=attn_mask_2d.device,
                                   dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
    padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1, 1, 1, seq_len)
    return padding_mask & causal  # (1, 1, seq_len, seq_len)

# === Stage1, Stage2, Stage3 ===

class Stage1(torch.nn.Module):
    def __init__(self, embed_tokens, rotary_emb, layers):
        super().__init__()
        self.embed  = embed_tokens
        self.rotary = rotary_emb
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, input_ids, decoder_attn_mask, position_ids):
        """
        input_ids: LongTensor (1, seq_len)
        decoder_attn_mask: BoolTensor (1, 1, seq_len, seq_len)
        position_ids: LongTensor (1, seq_len)
        """
        hidden_states = self.embed(input_ids)                      # (1, seq_len, hidden_size)
        cos, sin = self.rotary(hidden_states, position_ids)        # RoPE: cos и sin
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=decoder_attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin)
            )[0]
        return hidden_states  # (1, seq_len, hidden_size)

class Stage2(torch.nn.Module):
    def __init__(self, rotary_emb, layers):
        super().__init__()
        self.rotary = rotary_emb
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states, decoder_attn_mask, position_ids):
        """
        hidden_states:    (1, seq_len, hidden_size) из Stage1
        decoder_attn_mask: (1, 1, seq_len, seq_len) BoolTensor
        position_ids:      (1, seq_len) LongTensor
        """
        cos, sin = self.rotary(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=decoder_attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin)
            )[0]
        return hidden_states  # (1, seq_len, hidden_size)

class Stage3(torch.nn.Module):
    def __init__(self, rotary_emb, layers, final_norm, lm_head):
        super().__init__()
        self.rotary  = rotary_emb
        self.layers  = torch.nn.ModuleList(layers)
        self.norm    = final_norm
        self.lm_head = lm_head

    def forward(self, hidden_states, decoder_attn_mask, position_ids):
        """
        hidden_states:    (1, seq_len, hidden_size) из Stage2
        decoder_attn_mask: (1, 1, seq_len, seq_len) BoolTensor
        position_ids:      (1, seq_len) LongTensor
        """
        cos, sin = self.rotary(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=decoder_attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin)
            )[0]
        hidden_states = self.norm(hidden_states)      # LayerNorm
        logits = self.lm_head(hidden_states)          # (1, seq_len, vocab_size)
        return logits

# === Константа, обозначающая последнюю стадию ===

LAST_STAGE = 2

# === Класс PartitionedQwen2 с универсальным методом forward ===

class PartitionedQwen2:
    """
    stage = 0, 1 или 2.
    • stage=0  → загружает stage1.pth + токенизатор. 
    • stage=1  → загружает stage2.pth (без токенизатора).
    • stage=2  → загружает stage3.pth + токенизатор.
    """

    def __init__(self,
                 stage: int,
                 model_name: str = "Qwen/Qwen2-0.5B",
                 parts_dir: str = "."):
        self.stage = stage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if stage in (0, LAST_STAGE):
            self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = None

        path = f"{parts_dir}/stage{stage+1}.pth"
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.to(self.device).eval()

    def forward(self, inputs: dict) -> dict:
        # ===== STAGE 0 =====
        if self.stage == 0:
            # Если есть "prompt", это первый вызов для данного таска
            if "prompt" in inputs:
                prompt = inputs["prompt"]
                encoding = self.tokenizer(prompt, return_tensors="pt")
                input_ids = encoding.input_ids.to(self.device)       # (1, seq_len)
                id_list = input_ids.squeeze().tolist()               # [id0, id1, ...]
                seq_len = input_ids.size(1)

                attn_mask_1d = torch.ones((1, seq_len), device=self.device, dtype=torch.long)
                decoder_attn_mask = build_decoder_attention_mask(attn_mask_1d)
                position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    hidden1 = self.model(input_ids, decoder_attn_mask, position_ids)  # (1, seq_len, hidden_size)

                meta = tensor_to_base64(hidden1)
                hidden_meta = {"b64": meta["b64"], "dtype": meta["dtype"], "shape": meta["shape"]}

                return {
                    "hidden_meta":   hidden_meta,
                    "generated_ids": id_list
                }

            # Иначе это повторный вызов Stage 0 с уже накопленным списком ID
            elif "generated_ids" in inputs:
                gen_ids = inputs["generated_ids"]
                input_ids = torch.tensor([gen_ids], dtype=torch.long, device=self.device)  # (1, seq_len)
                seq_len = input_ids.size(1)

                attn_mask_1d = torch.ones((1, seq_len), device=self.device, dtype=torch.long)
                decoder_attn_mask = build_decoder_attention_mask(attn_mask_1d)
                position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    hidden1 = self.model(input_ids, decoder_attn_mask, position_ids)

                meta = tensor_to_base64(hidden1)
                hidden_meta = {"b64": meta["b64"], "dtype": meta["dtype"], "shape": meta["shape"]}

                return {
                    "hidden_meta":   hidden_meta,
                    "generated_ids": gen_ids
                }

            else:
                raise RuntimeError("Stage=0: нужно передать либо 'prompt', либо 'generated_ids'.")

        # ===== STAGE 1 =====
        elif self.stage == 1:
            if "hidden_meta" not in inputs:
                raise RuntimeError("Stage=1: требуется 'hidden_meta'.")
            hidden_meta_in = inputs["hidden_meta"]
            hidden1 = base64_to_tensor({
                "b64":   hidden_meta_in["b64"],
                "dtype": hidden_meta_in["dtype"],
                "shape": hidden_meta_in["shape"]
            }).to(self.device)  # (1, seq_len, hidden_size)

            seq_len = hidden1.size(1)
            attn_mask_1d = torch.ones((1, seq_len), device=self.device, dtype=torch.long)
            decoder_attn_mask = build_decoder_attention_mask(attn_mask_1d)
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

            with torch.no_grad():
                hidden2 = self.model(hidden1, decoder_attn_mask, position_ids)  # (1, seq_len, hidden_size)

            meta2 = tensor_to_base64(hidden2)
            # Раньше мы возвращали hidden2_meta, теперь просто hidden_meta
            hidden_meta = {"b64": meta2["b64"], "dtype": meta2["dtype"], "shape": meta2["shape"]}

            return {
                "hidden_meta": hidden_meta
            }

        # ===== STAGE 2 =====
        elif self.stage == LAST_STAGE:
            if "hidden_meta" not in inputs or "generated_ids" not in inputs:
                raise RuntimeError("Stage=2: требуются 'hidden_meta' и 'generated_ids'.")
            hidden_meta_in = inputs["hidden_meta"]
            gen_ids = inputs["generated_ids"]

            hidden2 = base64_to_tensor({
                "b64":   hidden_meta_in["b64"],
                "dtype": hidden_meta_in["dtype"],
                "shape": hidden_meta_in["shape"]
            }).to(self.device)  # (1, seq_len, hidden_size)

            seq_len = hidden2.size(1)
            attn_mask_1d = torch.ones((1, seq_len), device=self.device, dtype=torch.long)
            decoder_attn_mask = build_decoder_attention_mask(attn_mask_1d)
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits_all = self.model(hidden2, decoder_attn_mask, position_ids)  # (1, seq_len, vocab_size)

            logits_last = logits_all[:, -1, :]
            next_id = torch.argmax(logits_last, dim=-1).squeeze().item()
            next_str = self.tokenizer.decode(next_id, skip_special_tokens=True)

            new_ids = gen_ids + [next_id]
            return {
                "next_token_id":  next_id,
                "next_token_str": next_str,
                "generated_ids":  new_ids
            }

        else:
            raise RuntimeError(f"Недопустимый stage={self.stage}.")