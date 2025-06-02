import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer
import torch
import base64
import numpy as np

# Класс, который возвращает сразу и base64, и форму тензора
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
        "b64": b64,
        "dtype": str(array.dtype),               # например, "float32"
        "shape": list(array.shape),             # [1, seq_len, hidden_size]
    }


def base64_to_tensor(meta: dict) -> torch.Tensor:
    """
    Ожидает dictionary из tensor_to_base64_with_meta:
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



class Stage1(torch.nn.Module):
    def __init__(self, embed_tokens, rotary_emb, layers):
        super().__init__()
        self.embed      = embed_tokens
        self.rotary     = rotary_emb
        self.layers     = torch.nn.ModuleList(layers)

    def forward(self, input_ids, decoder_attn_mask, position_ids):
        hidden_states = self.embed(input_ids)                      # (1, seq_len, hidden_size)
        cos, sin = self.rotary(hidden_states, position_ids)       # RoPE cos и sin
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=decoder_attn_mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin)
            )[0]

        
        return hidden_states  # (1, seq_len, hidden_size)


def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
    """
    Формирует 4D-каузальную маску из 2D padding-маски (все единицы, если нет паддинга).
    attn_mask_2d: (1, seq_len), LongTensor из единиц.
    Возвращает BoolTensor (1, 1, seq_len, seq_len).
    """
    seq_len = attn_mask_2d.size(1)
    causal = torch.tril(torch.ones((seq_len, seq_len), device=attn_mask_2d.device, dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
    padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
    return padding_mask & causal  # (1,1,seq_len,seq_len)


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

def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
    """
    Формирует 4D-каузальную маску (аналогично Stage1).
    attn_mask_2d: (1, seq_len), LongTensor из единиц.
    """
    seq_len = attn_mask_2d.size(1)
    causal = torch.tril(torch.ones((seq_len, seq_len), device=attn_mask_2d.device, dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(1)  # (1,1,seq_len,seq_len)
    padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
    return padding_mask & causal  # (1,1,seq_len,seq_len)



class Stage3(torch.nn.Module):
    def __init__(self, rotary_emb, layers, final_norm, lm_head):
        super().__init__()
        self.rotary   = rotary_emb
        self.layers   = torch.nn.ModuleList(layers)
        self.norm     = final_norm
        self.lm_head  = lm_head

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

def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
    """
    Формирует 4D-каузальную маску (аналогично Stage1/Stage2).
    attn_mask_2d: (1, seq_len), LongTensor из единиц.
    """
    seq_len = attn_mask_2d.size(1)
    causal = torch.tril(torch.ones((seq_len, seq_len), device=attn_mask_2d.device, dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(1)  # (1,1,seq_len,seq_len)
    padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
    return padding_mask & causal  # (1,1,seq_len,seq_len)


LAST_STAGE = 2

class PartitionedQwen2:
    def __init__(self, stage: int, model_name="Qwen/Qwen2-0.5B", parts_dir="."):
        self.stage = stage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sessions = {}  # task_id → session state

        if stage in (0, LAST_STAGE):
            self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = None

        path = f"{parts_dir}/stage{stage+1}.pth"
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.to(self.device).eval()

    def forward(self, inp, task_id):
        if self.stage == 0:
            prompt = inp
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            seq_len = input_ids.size(1)

            attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
            dec_attn_mask = build_decoder_attention_mask(attn_mask)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                hidden1 = self.model(input_ids, dec_attn_mask, pos_ids)

            # Сохраняем всё в сессию
            self.sessions[task_id] = {
                "input_ids": input_ids,
                "text": prompt,
                "hidden1": hidden1,
            }
            return  # Stage 0 ничего не возвращает

        elif self.stage == 1:
            session = self.sessions.get(task_id)
            if session is None:
                raise ValueError(f"No session found for task_id={task_id}")
            hidden1 = session["hidden1"]
            seq_len = hidden1.size(1)

            attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
            dec_attn_mask = build_decoder_attention_mask(attn_mask)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                hidden2 = self.model(hidden1, dec_attn_mask, pos_ids)

            # Обновляем hidden2
            session["hidden2"] = hidden2
            return  # Stage 1 ничего не возвращает

        else:
            session = self.sessions.get(task_id)
            if session is None:
                raise ValueError(f"No session found for task_id={task_id}")
            hidden2 = session["hidden2"]
            input_ids = session["input_ids"]
            seq_len = hidden2.size(1)

            attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
            dec_attn_mask = build_decoder_attention_mask(attn_mask)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits_all = self.model(hidden2, dec_attn_mask, pos_ids)

            logits_last = logits_all[:, -1, :]
            next_token_id = torch.argmax(logits_last, dim=-1)
            next_token_int = next_token_id.item()
            next_token_str = self.tokenizer.decode(next_token_int, skip_special_tokens=True)

            # Обновляем state
            next_token_tensor = torch.tensor([[next_token_int]], device=self.device)
            session["input_ids"] = torch.cat([input_ids, next_token_tensor], dim=1)
            session["text"] += next_token_str

            # Заново считаем hidden1 для нового текста
            new_input = session["text"]
            input_ids = self.tokenizer(new_input, return_tensors="pt").input_ids.to(self.device)
            session["input_ids"] = input_ids
            seq_len = input_ids.size(1)

            attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
            dec_attn_mask = build_decoder_attention_mask(attn_mask)
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                hidden1 = self.model.embed(input_ids)  # только embed
            session["hidden1"] = hidden1

            return next_token_str
