from transformers import Trainer
import torch.nn.functional as F
import torch
from .utils import compute_log_likelihood
from typing import List, Optional, Tuple, Union

from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class SPAGTrainingArguments(TrainingArguments):
    lm_sft_coeff: float = field(default=0.1, metadata={"help": "Weight for supervised fine-tuning loss"})
    lm_kl_coeff: float = field(default=0.01, metadata={"help": "Weight for KL divergence penalty"})
    clip_range: float = field(default=0.2, metadata={"help": "Clip range for importance ratio"})

class SPAGTrainer(Trainer):
    """
    Trainer for Offline Policy Optimization, which combines PPO-style objective
    with supervised fine-tuning. Reference: https://github.com/Linear95/SPAG/blob/main/trainers.py
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss for Offline PPO training.

        Args:
            model: The model being trained
            inputs: The inputs for the model
            return_outputs: Whether to return the model outputs along with the loss

        Returns:
            Loss or (loss, outputs) if return_outputs is True
        """
        # Get model output
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )

        # Get reference model output (without gradient tracking)
        with torch.no_grad():
            ref_outputs = model.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
            ref_logprob = compute_log_likelihood(ref_outputs.logits, inputs['labels']).detach()

        # Compute log likelihood of current model
        logprob = compute_log_likelihood(outputs.logits, inputs['labels'])

        # Compute KL divergence between current and reference model
        kl_div = (logprob - ref_logprob)

        # Compute importance ratio (policy ratio)
        importance_ratio = (logprob - ref_logprob).exp()
        importance_ratio_clipped = torch.clip(
            importance_ratio,
            1 - self.args.clip_range,
            1 + self.args.clip_range
        )

        # Compute advantages (reward - KL penalty)
        advantages = inputs['reward'] - self.args.lm_kl_coeff * kl_div

        # Compute PPO-style clipped objective
        ppo_loss = -torch.minimum(
            advantages * importance_ratio,
            advantages * importance_ratio_clipped
        )

        # Separate samples into SFT and PPO
        sample_size = (1 - inputs['sft_mask']).sum()
        sft_size = inputs['sft_mask'].sum()

        # Compute SFT loss (only for SFT samples)
        sft_loss = 0
        if sft_size > 0:
            sft_loss = (-logprob * inputs['sft_mask'] * inputs['weights']).sum() / sft_size

        # Compute PPO loss (only for PPO samples)
        ppo_value = 0
        if sample_size > 0:
            ppo_value = (ppo_loss * (1 - inputs['sft_mask']) * inputs['weights']).sum() / sample_size

        # Combine losses with weighting
        weighted_loss = self.args.lm_sft_coeff * sft_loss + ppo_value

        # Log metrics to wandb
        if self.args.report_to and "wandb" in self.args.report_to:
            # Only log metrics on main process (rank 0)
            if self.args.local_rank == 0:
                import wandb
                wandb.log({
                    "train/sft_loss": sft_loss.item() if sft_size > 0 else 0,
                    "train/ppo_loss": ppo_value.item() if sample_size > 0 else 0,
                    "train/weighted_loss": weighted_loss.item(),
                    "train/kl_div_mean": kl_div.mean().item(),
                    "train/importance_ratio_mean": importance_ratio.mean().item(),
                    "train/advantages_mean": advantages.mean().item(),
                    "train/sft_sample_ratio": sft_size / (sft_size + sample_size) if (sft_size + sample_size) > 0 else 0,
                })

        return (weighted_loss, outputs) if return_outputs else weighted_loss