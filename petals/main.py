import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer
# from stages import Stage1, Stage2, Stage3

from copy import copy, deepcopy
from partitioned_models import PartitionedQwen2, LAST_STAGE

def build_decoder_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
    seq_len = attn_mask_2d.size(1)
    causal = torch.tril(
        torch.ones((seq_len, seq_len), device=attn_mask_2d.device, dtype=torch.bool)
    )
    causal = causal.unsqueeze(0).unsqueeze(1)  # (1,1,seq_len,seq_len)
    padding_mask = attn_mask_2d.unsqueeze(1).unsqueeze(2).to(torch.bool)  # (1,1,1,seq_len)
    return padding_mask & causal  # (1,1,seq_len,seq_len)


if __name__ == "__main__":
    device = torch.device("cpu")
    print("Initializing PartitionedQwen2 for stages 0, 1, 2 ...")
    # Создаём три объекта: p0 для stage = 0, p1 для stage = 1, p2 для stage = 2
    p0 = PartitionedQwen2(stage=0, model_name="Qwen/Qwen2-0.5B", parts_dir=".")
    p1 = PartitionedQwen2(stage=1, parts_dir=".")
    p2 = PartitionedQwen2(stage=LAST_STAGE, model_name="Qwen/Qwen2-0.5B", parts_dir=".")

    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    prompt = "Расскажи про погоду сегодня в Хельсинки пожалуйста."
    # Начнём со стадии 0: получим hidden1 для всего prompt
    hidden1 = p0.forward(prompt=prompt)  # (1, seq_len, hidden_size)
    generated_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # (1, seq_len)

    max_new_tokens = 50
    do_sample = True
    temperature = 1.0
    top_k = 50
    top_p = 0.9

    # Распечатать исходный prompt
    print(tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True), end="")
    aboba = []

    for _ in range(max_new_tokens):
        # Сохраняем текущее seq_len, построим attention_mask
        seq_len = generated_ids.size(1)
        attn_mask_1d = torch.ones((1, seq_len), dtype=torch.long, device=device)

        # Стадия 1: получаем hidden2
        hidden2 = p1.forward(
            hidden_states=hidden1,
            attention_mask=attn_mask_1d
        )  # (1, seq_len, hidden_size)

        # Стадия 2: получаем строковый токен (greedy)
        next_token_str = p2.forward(
            hidden_states=hidden2,
            attention_mask=attn_mask_1d
        )  # str

        # Если EOS или пусто — прерываемся
        if next_token_str == "" or next_token_str == tokenizer.eos_token:
            break
        aboba.append(copy(next_token_str))
        # Печатаем и добавляем токен в generated_ids
        print(next_token_str, end="", flush=True)
        next_id = tokenizer.encode(next_token_str, add_special_tokens=False)[0]
        new_token_tensor = torch.tensor([[next_id]], device=device)
        generated_ids = torch.cat([generated_ids, new_token_tensor], dim=1)

        # Чтобы на следующей итерации получить hidden1, 
        # передаём весь накопленный текст в p0:
        full_text = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
        hidden1 = p0.forward(prompt=full_text)
        

    print()  # перевод строки в конце
    print(aboba)
