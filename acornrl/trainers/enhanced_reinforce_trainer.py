import torch
import torch.nn as nn
from transformers import Trainer
import torch.nn.functional as F
# from .baseline import Baseline
# from .utils import compute_log_likelihood

class EnhancedReinforceTrainer(Trainer):
    def __init__(self, baseline_lr=1e-4, kl_coeff=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
    #     self.baseline = Baseline(self.model.config.hidden_size).to(self.model.device)
    #     self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr=baseline_lr)
    #     self.kl_coeff = kl_coeff

    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #     rewards = inputs.pop("reward")
    #     labels = inputs["labels"]

    #     outputs = model(**inputs)
    #     logits = outputs.logits

    #     log_probs = compute_log_likelihood(logits, labels)

    #     with torch.no_grad():
    #         baseline_preds = self.baseline(outputs.hidden_states[-1].detach()).squeeze(-1)

    #     advantages = rewards - baseline_preds

    #     pg_loss = -torch.mean(log_probs * advantages.detach())
    #     baseline_loss = F.mse_loss(baseline_preds, rewards)

    #     with torch.no_grad():
    #         ref_outputs = model.ref_model(**inputs)
    #         ref_log_probs = compute_log_likelihood(ref_outputs.logits, labels)

    #     kl_loss = self.kl_coeff * (log_probs - ref_log_probs).mean()

    #     total_loss = pg_loss + baseline_loss + kl_loss

    #     # Baseline update
    #     self.baseline_optimizer.zero_grad()
    #     baseline_loss.backward(retain_graph=True)
    #     self.baseline_optimizer.step()

    #     if return_outputs:
    #         outputs.loss = total_loss
    #         return total_loss, outputs
    #     else:
    #         return total_loss
