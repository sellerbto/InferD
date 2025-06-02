import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer

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

def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
    """
    Формирует 4D-каузальную маску из 2D padding-маски (все единицы).
    attn_mask_2d: (1, seq_len), LongTensor из единиц.
    Возвращает BoolTensor (1,1,seq_len,seq_len).
    """
    seq_len = attn_mask_2d.size(1)
    causal = torch.tril(torch.ones((seq_len, seq_len),
                                   device=attn_mask_2d.device,
                                   dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(1)            # (1,1,seq_len,seq_len)
    padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
    return padding_mask & causal                        # (1,1,seq_len,seq_len)


class PartitionedQwen2:
    def __init__(self, stage: int, model_name="Qwen/Qwen2-0.5B", parts_dir="."):
        """
        stage: 0, 1 или 2 (LAST_STAGE). Загружает только один этап:
          • stage=0  → загружается stage0.pth + токенизатор
          • stage=1  → загружается stage1.pth (нет токенизатора)
          • stage=2  → загружается stage2.pth + токенизатор
        parts_dir: папка, в которой лежат файлы stage0.pth, stage1.pth, stage2.pth.
        """
        self.stage = stage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Только для stage=0 и stage=2 нужен токенизатор:
        if stage in (0, LAST_STAGE):
            self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = None

        # Загружаем нужный .pth-файл
        path = f"stage{stage+1}.pth"
        # Указываем weights_only=False, чтобы PyTorch мог распаковать класс Stage*
        self.model = torch.load(path, map_location=self.device, weights_only=False)
        self.model.to(self.device).eval()

    def forward(self, **kwargs):
        if self.stage == 0:
            # -------------------- STAGE 0 --------------------
            prompt = kwargs.get("prompt", None)
            if prompt is None:
                raise ValueError("Для stage=0 необходимо передать аргумент prompt=str")
            # 1) Токенизуем prompt
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)  # (1, seq_len)
            seq_len = input_ids.size(1)

            # 2) Строим attention_mask (1D из единиц) и 4D каузальную маску
            attn_mask_1d = torch.ones((1, seq_len), dtype=torch.long, device=self.device)        # (1, seq_len)
            decoder_attn_mask = build_decoder_attention_mask(attn_mask_1d)                       # (1,1,seq_len,seq_len)

            # 3) Строим position_ids
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, seq_len)

            # 4) Запускаем stage0 → hidden1
            with torch.no_grad():
                hidden1 = self.model(input_ids, decoder_attn_mask, position_ids)  # (1, seq_len, hidden_size)
            return hidden1

        elif self.stage == 1:
            # -------------------- STAGE 1 --------------------
            hidden_states = kwargs.get("hidden_states", None)
            attention_mask = kwargs.get("attention_mask", None)  # ожидаем shape (1, seq_len)
            if hidden_states is None or attention_mask is None:
                raise ValueError("Для stage=1 необходимо передать hidden_states и attention_mask")
            seq_len = hidden_states.size(1)

            # 1) 4D каузальная маска
            decoder_attn_mask = build_decoder_attention_mask(attention_mask)  # (1,1,seq_len,seq_len)

            # 2) position_ids
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

            # 3) Запускаем stage1 → hidden2
            with torch.no_grad():
                hidden2 = self.model(hidden_states, decoder_attn_mask, position_ids)  # (1, seq_len, hidden_size)
            return hidden2

        else:
            # -------------------- STAGE 2 (LAST_STAGE) --------------------
            hidden_states = kwargs.get("hidden_states", None)
            attention_mask = kwargs.get("attention_mask", None)  # (1, seq_len)
            if hidden_states is None or attention_mask is None:
                raise ValueError("Для stage=2 необходимо передать hidden_states и attention_mask")
            seq_len = hidden_states.size(1)

            # 1) 4D каузальная маска
            decoder_attn_mask = build_decoder_attention_mask(attention_mask)  # (1,1,seq_len,seq_len)

            # 2) position_ids
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

            # 3) Запускаем stage2 → получаем логиты
            with torch.no_grad():
                logits_all = self.model(hidden_states, decoder_attn_mask, position_ids)  # (1, seq_len, vocab_size)

            # 4) Берём логиты последнего токена и выбираем greedy
            logits_last = logits_all[:, -1, :]                   # (1, vocab_size)
            next_token_id = torch.argmax(logits_last, dim=-1)    # (1,)
            next_token_int = next_token_id.squeeze().item()

            # 5) Декодируем токен в строку
            next_token_str = self.tokenizer.decode(next_token_int, skip_special_tokens=True)
            return next_token_str
