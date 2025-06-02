import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer
import base64
import numpy as np

from partitioned_models import PartitionedQwen2, LAST_STAGE

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

def tensor_to_base64(tensor: torch.Tensor) -> dict:
    """
    Кодирует torch.Tensor в dict:
      {
        "b64": "<байты закодированные в base64>",
        "dtype": "float32",
        "shape": [1, seq_len, hidden_size]
      }
    """
    array = tensor.detach().cpu().numpy()
    b64 = base64.b64encode(array.tobytes()).decode("utf-8")
    return {
        "b64": b64,
        "dtype": str(array.dtype),      # например, "float32"
        "shape": list(array.shape),     # [1, seq_len, hidden_size]
    }


def base64_to_tensor(meta: dict) -> torch.Tensor:
    """
    Обратная операция: из словаря, созданного tensor_to_base64, восстанавливает torch.Tensor той же формы.
    meta должно быть:
      {
        "b64": "<строка>",
        "dtype": "float32",
        "shape": [1, seq_len, hidden_size]
      }
    """
    b64  = meta["b64"]
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    raw  = base64.b64decode(b64)
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return torch.from_numpy(array).clone()


# =======================================
# Session‐Manager, обёртка над тремя стадиями
# =======================================

class Qwen2SessionManager:
    """
    Этот класс внутри себя держит три PartitionedQwen2 (stage0, stage1, stage2),
    а также словарь self.sessions, где сохраняются состояния (hidden1_b64, generated_ids) по каждому request_id.
    
    Публичные методы:
      - start(request_id: str, prompt: str) — инициализировать новую сессию.
      - generate_next(request_id: str) -> str — получить следующий токен (greedy) и обновить состояние.
    """

    def __init__(self,
                 model_name="Qwen/Qwen2-0.5B",
                 parts_dir=".",
                 device: str = None):
        """
        model_name: путь/имя для загрузки токенизатора (только в stage0 и stage2).
        parts_dir: папка, где лежат файлы stage1.pth, stage2.pth, stage3.pth и т.д.
        device: "cpu" или "cuda", либо None → автоопределение.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # создаём три экземпляра модели для трёх стадий
        self.p0 = PartitionedQwen2(stage=0, model_name=model_name, parts_dir=parts_dir)
        self.p1 = PartitionedQwen2(stage=1, parts_dir=parts_dir)
        self.p2 = PartitionedQwen2(stage=LAST_STAGE, model_name=model_name, parts_dir=parts_dir)

        # Токенизатор нужен, чтобы конвертировать текст ↔ токены
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

        # Словарь, где будут храниться состояния для каждой сессии
        # sessions[request_id] = {
        #     "generated_ids": Tensor (1, curr_seq_len),
        #     "hidden1_meta": {"b64":..., "dtype":..., "shape": [...]},
        #     "finished": bool
        # }
        self.sessions = {}

    def start(self, request_id: str, prompt: str):
        """
        Инициализирует новую сессию с ключом request_id:
          1. Токенизирует prompt → input_ids (1, seq_len0).
          2. Запускает stage0 (p0) → получает hidden1 (тензор).
          3. Кодирует hidden1 в base64+metadata.
          4. Сохраняет generated_ids и hidden1_meta в self.sessions[request_id].
          5. Помечает finished=False.
        """
        if request_id in self.sessions:
            raise ValueError(f"Сессия с request_id={request_id!r} уже существует")

        # 1) Токенизация
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.device)  # (1, seq_len0)

        # 2) Строим маски и position_ids для stage0 внутри PartitionedQwen2, 
        #    но так как p0.forward ждёт на вход «prompt», он сам токенизирует и строит всё.
        #    Можно напрямую вызвать p0.forward(prompt), но тогда мы не получим input_ids здесь.
        #    Поэтому сделаем так: используем напрямую ту же логику, что и p0, без дублирования кода.
        #
        #    Удобнее всего просто позвать p0.forward(prompt), 
        #    который вернёт нам «base64+meta» для hidden1.
        #
        hidden1_meta = self.p0.forward(prompt)  # это dict с ключами b64, dtype, shape

        # 3) Сохраняем в сессии:
        self.sessions[request_id] = {
            "generated_ids": input_ids,   # Tensor [1, seq_len0]
            "hidden1_meta":  hidden1_meta,
            "finished":       False
        }

    def generate_next(self, request_id: str) -> str:
        """
        Генерирует следующий токен для сеанса request_id (greedy):
          1. Берёт из self.sessions «hidden1_meta» → base64_to_tensor(hidden1_meta).
          2. Строит attention_mask_1d для current_seq_len.
          3. Запускает stage1 (p1) → получает hidden2_meta (base64+meta).
          4. Запускает stage2 (p2) → получает next_token_str.
          5. Если next_token_str == EOS, помечаем finished=True и возвращаем "".
             Иначе:
               a) добавляем next_token_id в generated_ids (расширяем по dim=1).
               b) создаём полный текст: tokenizer.decode(generated_ids).
               c) запускаем stage0 заново (p0.forward) с полным текстом → получаем новый hidden1_meta.
               d) сохраняем в сессии обновлённые generated_ids и hidden1_meta.
               e) возвращаем next_token_str.
        """
        if request_id not in self.sessions:
            raise KeyError(f"Нет сессии с request_id={request_id!r}")

        sess = self.sessions[request_id]
        if sess["finished"]:
            return ""  # уже закончили

        # 1) Извлекаем hidden1_meta → восстанавливаем Tensor hidden1
        hidden1_meta = sess["hidden1_meta"]
        hidden1 = base64_to_tensor(hidden1_meta).to(self.device)  # Tensor [1, seq_len, hidden_size]

        # 2) Строим attention_mask на основе current_seq_len
        generated_ids = sess["generated_ids"]  # Tensor [1, curr_seq_len]
        seq_len = generated_ids.size(1)
        attn_mask_1d = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        decoder_attn_mask = build_decoder_attention_mask(attn_mask_1d)  # BoolTensor [1,1,seq_len,seq_len]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

        # 3) Запускаем Stage1 → hidden2 (тензор)
        with torch.no_grad():
            hidden2 = self.p1.model(  # поскольку p1.forward ожидает base64_meta, но мы хотим вызвать model напрямую:
                hidden1,  # [1, seq_len, hidden_size]
                decoder_attn_mask,
                position_ids
            )[0]  # p1.forward возвращал тензор hidden2, но здесь self.p1.model(...) возвращает tuple[hidden, ...], поэтому [0].
        # Сразу упаковываем hidden2 в base64+meta:
        hidden2_meta = tensor_to_base64(hidden2)

        # 4) Запускаем Stage2 → получаем логиты и next_token_str
        #    Stage2 в PartitionedQwen2 внутри себя принимает base64_meta, 
        #    но нам удобнее вызвать модель напрямую, чтобы сразу получить logits.
        with torch.no_grad():
            logits_all = self.p2.model(
                hidden2,  # tensor
                decoder_attn_mask,
                position_ids
            )[0]  # аналогично, model(...) возвращает (logits, …), поэтому [0]
        logits_last = logits_all[:, -1, :]                     # (1, vocab_size)
        next_token_id = torch.argmax(logits_last, dim=-1).item()  # int
        next_token_str = self.tokenizer.decode(next_token_id, skip_special_tokens=True)

        # 5) Проверяем на EOS или пустую строку
        if next_token_str == "" or next_token_str == self.tokenizer.eos_token:
            sess["finished"] = True
            return ""

        # Иначе — добавляем токен в generated_ids и пересчитываем hidden1 «с нуля»
        # а) Добавляем next_token_id
        new_token_tensor = torch.tensor([[next_token_id]], device=self.device)  # Shape (1,1)
        sess["generated_ids"] = torch.cat([generated_ids, new_token_tensor], dim=1)  # (1, seq_len+1)

        # б) Строим полный текст (prompt + все сгенерированные токены):
        full_ids = sess["generated_ids"].squeeze().tolist()
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)

        # в) Перезапускаем stage0 по полному тексту → получаем обновлённый hidden1_meta
        hidden1_meta_new = self.p0.forward(full_text)  # возвращает dict
        sess["hidden1_meta"] = hidden1_meta_new

        return next_token_str

    def is_finished(self, request_id: str) -> bool:
        """
        Помеха: проверить, завершился ли выбор следующего токена (достигли ли EOS).
        """
        if request_id not in self.sessions:
            raise KeyError(f"Нет сессии с request_id={request_id!r}")
        return self.sessions[request_id]["finished"]

    def reset(self, request_id: str):
        """
        Удаляет сессию из памяти (если хотите, чтобы можно было перезапустить с тем же ID).
        """
        if request_id in self.sessions:
            del self.sessions[request_id]
