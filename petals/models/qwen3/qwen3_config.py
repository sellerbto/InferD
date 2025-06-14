class Qwen3Config:
    # infra
    HF_REPO_ID = "yellooot/inferd-qwen3"
    # generation
    TEMPERATURE: float = 0.6
    TOP_K: int = 20
    TOP_P: float = 0.95
    EOS_TOKEN_ID: int = 151645
    # model
    HIDDEN_SIZE: int = 1024
    NUM_HIDDEN_LAYERS: int = 28
    NUM_ATTENTION_HEADS: int = 16
    VOCAB_SIZE: int = 151936
    MAX_POSITION_EMBEDDINGS: int = 40960
    HEAD_DIM: int = 128
    INTERMEDIATE_SIZE: int = 3072
    PAD_TOKEN_ID: int = 151643
    BOS_TOKEN_ID: int = 151643
    RMS_NORM_EPS: float = 1e-6
    HIDDEN_ACT: str = "silu"
    NUM_KEY_VALUE_HEADS: int = 8
    ATTENTION_BIAS: bool = False
    ATTENTION_TYPE: str = "full_attention"
    ROPE_THETA: int = 1000000
    TORCH_DTYPE: str = "bfloat16"
