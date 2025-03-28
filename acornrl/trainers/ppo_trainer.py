import torch
from transformers import Trainer
import torch.nn.functional as F
from .utils import compute_log_likelihood

class PPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards = inputs.pop("reward")
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        with torch.no_grad():
            ref_outputs = model.ref_model(**inputs)
            ref_log_probs = compute_log_likelihood(ref_outputs.logits, labels)

        current_log_probs = compute_log_likelihood(logits, labels)

        ratios = torch.exp(current_log_probs - ref_log_probs)
        clipped_ratios = torch.clamp(ratios, 1 - 0.2, 1 + 0.2)

        advantages = rewards - rewards.mean()

        ppo_loss = -torch.mean(torch.min(ratios * advantages, clipped_ratios * advantages))

        kl_loss = 0.01 * (current_log_probs - ref_log_probs).mean()

        total_loss = ppo_loss + kl_loss

        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs
        else:
            return total_loss
