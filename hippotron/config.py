# https://github.com/nathan-barry/RoBERTaDiffusion/blob/main/config.py
import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    # Model config
    MODEL_NAME: str = "UFNLP/gatortronS"
    MIN_PREFIX_LEN: int = 16
    MAX_PREFIX_LEN: int = 128
    MAX_LEN: int = 512

    # Inference config
    CONFIDENCE_THRESHOLD: float = 0.9
    TEMPERATURE: float = 0.8
    TOP_K: int = 50
    TOP_P: float = 0.95
    BLOCK_SIZE: int = 32
    NUM_STEPS: int = 256
    MAX_NEW_TOKENS: int | None = 128
    
    # Tokenizer config
    MASK_TOKEN_ID: int | None = 103
    PAD_TOKEN_ID: int | None = 0
    SPECIAL_IDS: list[int] = field(default_factory=list)

    # Training config
    BATCH_SIZE: int = 16
    OUTPUT_DIR: str = "weights"
    MAX_STEPS: int = 1000
    SAVE_STEPS: int = 500
    LOGGING_STEPS: int = 50
    SAVE_TOTAL_LIMIT: int = 1

    # Dataset
    DATASET_NAME: str = "AGBonnet/augmented-clinical-notes"
    TEXT_COLUMN: str = "note"
    TRAIN_SPLIT: str = "train"
    EVAL_SPLIT: str = "train"


def get_device() -> torch.device:
    """Select the best available device (MPS > CUDA > CPU).

    Returns:
        torch.device: Best available device
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("[INFO] Using MPS (Apple silicon) backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[INFO] Using CUDA backend")
        return torch.device("cuda")
    else:
        print("[INFO] Using CPU backend")
        return torch.device("cpu")