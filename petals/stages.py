# import torch
# from transformers import Qwen2Tokenizer

# class Stage1(torch.nn.Module):
#     def __init__(self, embed_tokens, rotary_emb, layers):
#         super().__init__()
#         self.embed      = embed_tokens
#         self.rotary     = rotary_emb
#         self.layers     = torch.nn.ModuleList(layers)

#     def forward(self, input_ids, decoder_attn_mask, position_ids):
#         hidden_states = self.embed(input_ids)                      # (1, seq_len, hidden_size)
#         cos, sin = self.rotary(hidden_states, position_ids)       # RoPE cos и sin
#         for layer in self.layers:
#             hidden_states = layer(
#                 hidden_states,
#                 attention_mask=decoder_attn_mask,
#                 position_ids=position_ids,
#                 position_embeddings=(cos, sin)
#             )[0]
#         return hidden_states  # (1, seq_len, hidden_size)


# def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
#     """
#     Формирует 4D-каузальную маску из 2D padding-маски (все единицы, если нет паддинга).
#     attn_mask_2d: (1, seq_len), LongTensor из единиц.
#     Возвращает BoolTensor (1, 1, seq_len, seq_len).
#     """
#     seq_len = attn_mask_2d.size(1)
#     causal = torch.tril(torch.ones((seq_len, seq_len), device=attn_mask_2d.device, dtype=torch.bool))
#     causal = causal.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
#     padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
#     return padding_mask & causal  # (1,1,seq_len,seq_len)


# class Stage2(torch.nn.Module):
#     def __init__(self, rotary_emb, layers):
#         super().__init__()
#         self.rotary = rotary_emb
#         self.layers = torch.nn.ModuleList(layers)

#     def forward(self, hidden_states, decoder_attn_mask, position_ids):
#         """
#         hidden_states:    (1, seq_len, hidden_size) из Stage1
#         decoder_attn_mask: (1, 1, seq_len, seq_len) BoolTensor
#         position_ids:      (1, seq_len) LongTensor
#         """
#         cos, sin = self.rotary(hidden_states, position_ids)
#         for layer in self.layers:
#             hidden_states = layer(
#                 hidden_states,
#                 attention_mask=decoder_attn_mask,
#                 position_ids=position_ids,
#                 position_embeddings=(cos, sin)
#             )[0]
#         return hidden_states  # (1, seq_len, hidden_size)

# def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
#     """
#     Формирует 4D-каузальную маску (аналогично Stage1).
#     attn_mask_2d: (1, seq_len), LongTensor из единиц.
#     """
#     seq_len = attn_mask_2d.size(1)
#     causal = torch.tril(torch.ones((seq_len, seq_len), device=attn_mask_2d.device, dtype=torch.bool))
#     causal = causal.unsqueeze(0).unsqueeze(1)  # (1,1,seq_len,seq_len)
#     padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
#     return padding_mask & causal  # (1,1,seq_len,seq_len)



# class Stage3(torch.nn.Module):
#     def __init__(self, rotary_emb, layers, final_norm, lm_head):
#         super().__init__()
#         self.rotary   = rotary_emb
#         self.layers   = torch.nn.ModuleList(layers)
#         self.norm     = final_norm
#         self.lm_head  = lm_head

#     def forward(self, hidden_states, decoder_attn_mask, position_ids):
#         """
#         hidden_states:    (1, seq_len, hidden_size) из Stage2
#         decoder_attn_mask: (1, 1, seq_len, seq_len) BoolTensor
#         position_ids:      (1, seq_len) LongTensor
#         """
#         cos, sin = self.rotary(hidden_states, position_ids)
#         for layer in self.layers:
#             hidden_states = layer(
#                 hidden_states,
#                 attention_mask=decoder_attn_mask,
#                 position_ids=position_ids,
#                 position_embeddings=(cos, sin)
#             )[0]
#         hidden_states = self.norm(hidden_states)      # LayerNorm
#         logits = self.lm_head(hidden_states)          # (1, seq_len, vocab_size)
#         return logits

# def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
#     """
#     Формирует 4D-каузальную маску (аналогично Stage1/Stage2).
#     attn_mask_2d: (1, seq_len), LongTensor из единиц.
#     """
#     seq_len = attn_mask_2d.size(1)
#     causal = torch.tril(torch.ones((seq_len, seq_len), device=attn_mask_2d.device, dtype=torch.bool))
#     causal = causal.unsqueeze(0).unsqueeze(1)  # (1,1,seq_len,seq_len)
#     padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
#     return padding_mask & causal  # (1,1,seq_len,seq_len)

