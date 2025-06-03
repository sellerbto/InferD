# main.py
from partitioned_models import PartitionedQwen2, Stage1, Stage2, Stage3, LAST_STAGE
from partitioned_session import Qwen2SessionManager

# main.py
from partitioned_models import PartitionedQwen2, LAST_STAGE

if __name__ == "__main__":
    p0 = PartitionedQwen2(stage=0, model_name="Qwen/Qwen2-0.5B", parts_dir=".")
    p1 = PartitionedQwen2(stage=1, parts_dir=".")
    p2 = PartitionedQwen2(stage=LAST_STAGE, model_name="Qwen/Qwen2-0.5B", parts_dir=".")

    prompt = "Привет, как дела?"
    gen_ids     = []
    hidden_meta = None

    print(prompt, end="", flush=True)

    max_new_tokens = 50
    for step in range(max_new_tokens):
        # —————— Stage 0 ——————
        if not gen_ids:
            out0 = p0.forward({"prompt": prompt})
        else:
            out0 = p0.forward({"generated_ids": gen_ids})

        hidden_meta = out0["hidden_meta"]
        gen_ids     = out0["generated_ids"]

        # —————— Stage 1 ——————
        out1 = p1.forward({"hidden_meta": hidden_meta})
        # Теперь Stage1 отдаёт {"hidden_meta": …}
        hidden_meta_stage2 = out1["hidden_meta"]

        # —————— Stage 2 ——————
        out2 = p2.forward({
            "hidden_meta": hidden_meta_stage2,
            "generated_ids": gen_ids
        })
        next_str = out2["next_token_str"]
        if next_str == "" or next_str == p2.tokenizer.eos_token:
            break

        print(next_str, end="", flush=True)
        gen_ids = out2["generated_ids"]

    print()  # переход на новую строку
    print("Генерация завершена.")
