# #!/usr/bin/env python3
# # split_qwen3.py

# import os
# import torch
# from transformers import AutoModelForCausalLM
# import torch.nn as nn

# MODEL_NAME = "Qwen/Qwen3-0.6B"
# SAVE_DIR = "./qwen3_parts"
# NUM_PARTS = 3

# os.makedirs(SAVE_DIR, exist_ok=True)

# # 1) Загружаем полную модель
# print(f"Loading full model {MODEL_NAME}...")
# full: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# full.eval()

# # 2) Берём компоненты
# #   - эмбеддинги и позиционные эмбеддинги
# #   - список трансформер-блоков
# #   - финальную нормализацию и голову
# embed_tokens     = full.model.embed_tokens
# embed_positions  = getattr(full.model, "embed_positions", None)  # если есть
# blocks           = full.model.layers  # обычно list(Module)
# final_ln         = getattr(full.model, "final_layer_norm", None)
# lm_head          = full.lm_head

# total_blocks = len(blocks)
# p1 = total_blocks // NUM_PARTS
# p2 = 2 * total_blocks // NUM_PARTS

# print(f"Total transformer blocks: {total_blocks}")
# print(f"Split indices: 0..{p1-1}, {p1}..{p2-1}, {p2}..{total_blocks-1}")


# class Part0(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Повторяем оригинальную иерархию имен
#         self.embed_tokens    = embed_tokens
#         if embed_positions is not None:
#             self.embed_positions = embed_positions
#         self.layers = torch.nn.ModuleList(blocks[:p1])

#     def forward(self, input_ids, attention_mask=None):
#         x = self.embed_tokens(input_ids)
#         if hasattr(self, "embed_positions"):
#             x = x + self.embed_positions(input_ids)
#         for block in self.layers:
#             # каждый block возвращает Tuple: (hidden_states, ...)
#             x = block(x, attention_mask=attention_mask)[0]
#         return x

# class Part1(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = torch.nn.ModuleList(blocks[p1:p2])

#     def forward(self, hidden_states, attention_mask=None):
#         x = hidden_states
#         for block in self.layers:
#             x = block(x, attention_mask=attention_mask)[0]
#         return x

# class Part2(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = torch.nn.ModuleList(blocks[p2:])
#         if final_ln is not None:
#             self.final_layer_norm = final_ln
#         self.lm_head = lm_head

#     def forward(self, hidden_states, attention_mask=None):
#         x = hidden_states
#         for block in self.layers:
#             x = block(x, attention_mask=attention_mask)[0]
#         if hasattr(self, "final_layer_norm"):
#             x = self.final_layer_norm(x)
#         logits = self.lm_head(x)
#         return logits

# # 4) Инстанцируем и сохраняем state_dict для каждой части
# parts = [Part0(), Part1(), Part2()]
# for idx, part in enumerate(parts):
#     fn = os.path.join(SAVE_DIR, f"part{idx}.pt")
#     # Сохраняем на CPU, чтобы не требовать доступ к GPU при загрузке
#     part_cpu = part.cpu()
#     torch.save(part_cpu.state_dict(), fn)
#     print(f"Saved part {idx} → {fn}")
