# split_qwen.py
# ────────────────────────────────────────────────────────────────────────────
# Скрипт:
# 1. Скачивает из HuggingFace модель Qwen/Qwen2-0.5B (на CPU).
# 2. Делит её на три «Stage»-модуля (по 8 слоёв в каждом).
# 3. Сохраняет каждый Stage в отдельный файл: stage1.pth, stage2.pth, stage3.pth.
#
# Запуск:
#   python split_qwen.py
# После этого в текущей папке появятся три файла:
#   - stage1.pth
#   - stage2.pth
#   - stage3.pth
# ────────────────────────────────────────────────────────────────────────────

import torch
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM

# 1. Загрузка модели (на CPU)
device = torch.device("cpu")
print("Скачиваем Qwen/Qwen2-0.5B и выгружаем на CPU...")
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-0.5B")
full_model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to(device)
full_model.eval()

# 2. Извлекаем основные модули из full_model
embed_tokens = full_model.model.embed_tokens    # nn.Embedding
rotary_emb    = full_model.model.rotary_emb     # объект RoPE
all_layers    = full_model.model.layers         # список из 24 Qwen2DecoderLayer
final_norm    = full_model.model.norm           # LayerNorm
lm_head       = full_model.lm_head              # линейный выходной слой (vocab_size)

num_layers = len(all_layers)                    # 24
cut1 = num_layers // 3                          # 8
cut2 = 2 * num_layers // 3                      # 16

layers_stage1 = all_layers[:cut1]               # слои 0..7
layers_stage2 = all_layers[cut1:cut2]           # слои 8..15
layers_stage3 = all_layers[cut2:]               # слои 16..23

# 3. Описываем классы Stage1, Stage2, Stage3 точно так же, как будем загружать их в inference-скриптах
class Stage1(torch.nn.Module):
    def __init__(self, embed_tokens, rotary_emb, layers):
        super().__init__()
        self.embed      = embed_tokens
        self.rotary     = rotary_emb
        self.layers     = torch.nn.ModuleList(layers)

    def forward(self, input_ids, decoder_attn_mask, position_ids):
        """
        input_ids:        (1, seq_len) LongTensor
        decoder_attn_mask: (1, 1, seq_len, seq_len) BoolTensor
        position_ids:     (1, seq_len) LongTensor
        """
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


class Stage2(torch.nn.Module):
    def __init__(self, rotary_emb, layers):
        super().__init__()
        self.rotary = rotary_emb
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states, decoder_attn_mask, position_ids):
        """
        hidden_states:     (1, seq_len, hidden_size) из Stage1
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
        self.rotary   = rotary_emb
        self.layers   = torch.nn.ModuleList(layers)
        self.norm     = final_norm
        self.lm_head  = lm_head

    def forward(self, hidden_states, decoder_attn_mask, position_ids):
        """
        hidden_states:     (1, seq_len, hidden_size) из Stage2
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
        hidden_states = self.norm(hidden_states)      # финальный LayerNorm
        logits = self.lm_head(hidden_states)          # (1, seq_len, vocab_size)
        return logits


# 4. Собираем экземпляры модулей
print("Создаём экземпляры Stage1, Stage2, Stage3 ...")
stage1 = Stage1(embed_tokens, rotary_emb, layers_stage1)
stage2 = Stage2(rotary_emb, layers_stage2)
stage3 = Stage3(rotary_emb, layers_stage3, final_norm, lm_head)

# Устанавливаем каждый в eval() (хотя при сохранении это не обязательно)
stage1.eval()
stage2.eval()
stage3.eval()

# 5. Сохраняем каждый Stage в отдельный файл
print("Сохраняем stage1.pth, stage2.pth, stage3.pth ...")
torch.save(stage1, "stage1.pth")
torch.save(stage2, "stage2.pth")
torch.save(stage3, "stage3.pth")

print("Готово! Три файла сохранены:")
print("  • stage1.pth")
print("  • stage2.pth")
print("  • stage3.pth")
