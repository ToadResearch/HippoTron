# https://huggingface.co/docs/transformers/en/trainer?ckpt=latest+checkpoint&nodes=multi-node#customize
# https://github.com/ZHZisZZ/dllm/blob/main/dllm/core/trainers/mdlm.py
# https://github.com/nathan-barry/RoBERTaDiffusion/blob/main/finetune.py
import torch
import torch.nn.functional as F
from transformers import Trainer

from config import Config


class BERTdLLMTrainer(Trainer):
    def __init__(self, *args, config: Config | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or Config()
        # cosine scheduler with normalized time t ∈ [0, 1]. (1-t) instead of t to start at 1 and end at 0.
        # smooth shape that decreases fast early on and then eases into 0 as opposed to a linear decrease of y = -x+1
        self.scheduler = lambda t: 1 - torch.cos((torch.pi/2) * (1-t))
        self.d_scheduler = lambda t: -torch.pi/2 * torch.sin((torch.pi/2) * (1-t))
        self.eps = 1e-3

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        
        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()
        
        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pad_token_id = self.config.PAD_TOKEN_ID
        mask_token_id = self.config.MASK_TOKEN_ID
        special_ids = self.config.SPECIAL_IDS

        # read tokenized input
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = (input_ids != pad_token_id).long()
        B, L = input_ids.shape

        # for each trajectory in the batch, sample a random timestep t ∈ [ε, 1)
        t = self.eps + (1 - self.eps) * torch.rand(B, device=input_ids.device) # (B,)

        # add new col dim, so we have one col with masking probability for
        # each sample in the batch. then copy this prob across each
        # row, so that each token in the sequence has the same masking prob
        # (B,) -- unsqueeze(1) --> (B, 1) -- expand(B, L) --> (B, L)
        p_mask = 1 - self.scheduler(t).unsqueeze(1).expand(B,L) # (B, L)

        # do not mask special or prompt tokens
        is_special = torch.zeros_like(input_ids, dtype=torch.bool) # (B, L)
        for s_id in special_ids:
            is_special |= input_ids == s_id # in-place OR op to modify is_special mask

        # maskable tokens can be predicted
        maskable = (attention_mask == 1) & (~is_special) & (labels != -100)

        # for maskable idxs, mask tokens with prob < p_mask
        masked_idx = maskable & (torch.rand((B,L), device=input_ids.device) < p_mask)
        noised_input_ids = torch.where(
            masked_idx, mask_token_id, input_ids
        )
        
        # forward pass
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # if there are no masked tokens, return 0 loss
        if not masked_idx.any():
            return (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0

        # weight CE loss per token with w(t) = - α'(t) / (1 - α(t))
        # prefactor from eq. 11 in https://arxiv.org/pdf/2406.07524
        loss_weights = (
            - self.d_scheduler(t) / (1 - self.scheduler(t) + 1e-6)
        ).unsqueeze(1).repeat(1, L) # (B,) --> (B,1) --> (B, L), repeat each sample weight across all its tokens

        # get CE loss of masked tokens
        # because reduction="none", we don't reduce to a single scalar value yet
        assert(input_ids[masked_idx]==labels[masked_idx]).all()
        token_loss = F.cross_entropy(
            logits[masked_idx], input_ids[masked_idx], reduction="none",
        ) # (num_masked_tokens,)
        token_loss *= loss_weights[masked_idx] # (num_masked_tokens,)

        # sum across each sample to get the number of tokens being predicted
        effective_lengths = maskable.sum(dim=1, keepdim=True).expand(B,L)
        # normalize each samples' loss with the num tokens being predicted
        # then sum all losses across samples, and normalize with num samples in the batch, B
        loss = torch.sum(token_loss/effective_lengths[masked_idx])/B

        return (loss, outputs) if return_outputs else loss
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortronS")
    train_config = Config(
        MASK_TOKEN_ID=tokenizer.mask_token_id,
        PAD_TOKEN_ID=tokenizer.pad_token_id,
        SPECIAL_IDS=tokenizer.all_special_ids
    )


