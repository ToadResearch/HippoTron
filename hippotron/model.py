# https://github.com/ZHZisZZ/dllm/blob/main/dllm/core/samplers/mdlm.py
# https://github.com/nathan-barry/RoBERTaDiffusion/blob/main/inference.py
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch 
from typing import Optional

from config import Config, get_device


class BERTdLLM:
    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        self.device = get_device() if device is None else device
        self.model = AutoModelForMaskedLM.from_pretrained(config.OUTPUT_DIR).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)

        self.schedule = torch.linspace(1, 0, config.NUM_STEPS) # linear scheduler from 1 to 0
    
    def tokenize(self, text: str):
        return self.tokenizer(
            text,
            max_length=self.config.MAX_LEN,
            truncation=False,
            padding=False,
            return_tensors="pt"
        )

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(
            ids, 
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )

    def preprocess(self, x: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        mask_token_id = self.tokenizer.mask_token_id
        max_prefix_length = self.config.MAX_PREFIX_LEN
        max_length = self.config.MAX_LEN
        
        # tokenize input
        encoded = self.tokenize(x)
        input_ids_prompt = encoded["input_ids"].squeeze(0)

        # make sure the prompt stays under the max prefix size
        prompt_length = input_ids_prompt.shape[0]
        prompt_length = min(prompt_length, max_prefix_length)

        # create array of mask tokens to be generated, and replace beginning with prompt
        current_ids = torch.full((1, max_length), fill_value=mask_token_id, dtype=torch.long)
        current_ids[0, :prompt_length] = input_ids_prompt[:prompt_length].clone()

        current_attention = torch.ones((1, max_length), dtype=torch.long)

        return current_ids.to(self.device), current_attention.to(self.device), prompt_length

    def postprocess(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def sample(
            logits: torch.Tensor,
            top_k: int = 0,
            top_p: float = 0.0,
            filter_val: float = -float("Inf"),
        ) -> torch.Tensor:
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
        # https://github.com/nathan-barry/RoBERTaDiffusion/blob/fb9309d2f2c962c418c68d4c421ff34d7184f5cf/inference.py#L48
    
    @torch.no_grad()
    def generate(self, text: str, save_history: bool = False):
        mask_token_id = self.config.MASK_TOKEN_ID

        # process the prompt and set up masked array
        initial_ids, attention_mask, prompt_length = self.preprocess(text)
        x = initial_ids.clone()
        B, L = x.shape

        # non-prompt tokens are masked
        masked_idx = (x == mask_token_id)
        
        history = []
        if save_history:
            history.append(self.decode(x[0])[3:]) # only save first in batch

        step = 0
        while masked_idx.any():
            step += 1

            with torch.no_grad():
                outputs = self.model(input_ids=x, attention_mask=attention_mask)
                logits = outputs.logits
            
            probs = torch.softmax(logits / self.config.TEMPERATURE, dim=-1)
            conf, pred = torch.max(probs, dim=-1)

            above_threshold = (
                conf > self.config.CONFIDENCE_THRESHOLD
            ) & masked_idx

            for i in range(B):
                if masked_idx[i].any() and not above_threshold[i].any():
                    masked_conf = conf[i].clone()
                    masked_conf[~masked_idx[i]] = -float("inf")
                    best_idx = torch.argmax(masked_conf)
                    above_threshold[i, best_idx] = True

            x = torch.where(above_threshold, pred, x)
            masked_idx = masked_idx & (~above_threshold)

            if save_history:
                history.append(self.decode(x[0])[3:])

        # in this case, we dont want to just sample the last token / batch
        # ids = self.sample(logits[-1, : , :], top_k=50, top_p=0.95)
        # return self.tokenizer.decode(ids, skip_special_tokens=True)
        return x, history

if __name__ == "__main__": 
    config = Config()
    model = BERTdLLM(config)
    ids, history = model.generate("23 year old male with a history of hypertension and diabetes presents with chest pain.", save_history=True)
    print(model.decode(ids[0]))
