"""
accelerate config
accelerate launch --num_processes=8 train.py
"""
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments

import accelerate
from datasets import load_dataset

from config import Config
from trainer import BERTdLLMTrainer


class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, config: Config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a batch of features with tokenization and dynamic masking."""
        # Extract text from features
        # Handle both dict-like and example-like features
        texts = []
        for f in features:
            if isinstance(f, dict):
                # fallbacks for the text column
                texts.append(f.get(self.config.TEXT_COLUMN, f.get("text", f.get("content", ""))))
            else:
                # IterableDataset items might be Example objects
                texts.append(f[self.config.TEXT_COLUMN] if self.config.TEXT_COLUMN in f else str(f))

        # Tokenize all texts to exactly MAX_LEN
        encoded = self.tokenizer(
            texts,
            max_length=self.config.MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        batch_input_ids = encoded["input_ids"]
        batch_attention = encoded["attention_mask"]

        B, L = batch_input_ids.shape

        # Clone input_ids to create labels
        labels = batch_input_ids.clone()

        # get a random prefix length for each example in the batch
        low, high = self.config.MIN_PREFIX_LEN, self.config.MAX_PREFIX_LEN
        assert low <= high, "MIN_PREFIX_LEN must be less than or equal to MAX_PREFIX_LEN"
        assert high < L, "MAX_PREFIX_LEN must be less than the sequence length"
        prefix_lengths = torch.randint(
            low, high+1, size=(B,)
        ).unsqueeze(1) # (B,) -> (B, 1)

        # copy down the row of idxs across all samples in batch
        # (L,) -> (1, L) -> (B, L)
        pos_idxs = torch.arange(L).unsqueeze(0).expand(B, L)
        is_prefix = pos_idxs < prefix_lengths # (B, L) < (B, 1)
        labels[is_prefix] = -100 # ignore prefix tokens

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": labels,
        }

        
if __name__ == "__main__":
    config = Config()
    
    print(f"[INFO] Loading model from {config.MODEL_NAME}...")
    model = AutoModelForMaskedLM.from_pretrained(config.MODEL_NAME)
    
    print(f"[INFO] Loading tokenizer from {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    config.MASK_TOKEN_ID = tokenizer.mask_token_id
    config.PAD_TOKEN_ID = tokenizer.pad_token_id
    config.SPECIAL_IDS = tokenizer.all_special_ids

    print(f"[INFO] Loading dataset from {config.DATASET_NAME}...")
    dataset = load_dataset(config.DATASET_NAME, streaming=True, trust_remote_code=True)
    
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        overwrite_output_dir=True,
        max_steps=config.MAX_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    print("[INFO] Initializing trainer...")
    trainer = BERTdLLMTrainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=dataset[config.TRAIN_SPLIT],
        eval_dataset=dataset[config.EVAL_SPLIT],
        data_collator=DataCollator(tokenizer, config),
    )

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    trainer.train()

    # Save model and tokenizer
    print(f"\n[INFO] Saving model to {config.OUTPUT_DIR}...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    print("[SUCCESS] Training complete!")