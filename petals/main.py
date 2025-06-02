# main.py
from partitioned_models import PartitionedQwen2, Stage1, Stage2, Stage3, LAST_STAGE
from partitioned_session import Qwen2SessionManager

if __name__ == "__main__":
    print("Initializing PartitionedQwen2...")
    p0 = PartitionedQwen2(stage=0, model_name="Qwen/Qwen2-0.5B", parts_dir=".")
    p1 = PartitionedQwen2(stage=1, parts_dir=".")
    p2 = PartitionedQwen2(stage=LAST_STAGE, model_name="Qwen/Qwen2-0.5B", parts_dir=".")

    task_id = "generation-42"
    prompt = "Привет."
    p0.forward(prompt, task_id=task_id)

    print(prompt, end="", flush=True)
    for _ in range(50):
        p1.forward(None, task_id=task_id)
        next_token = p2.forward(None, task_id=task_id)
        if not next_token or next_token == p2.tokenizer.eos_token:
            break
        print(next_token, end="", flush=True)

    print()
