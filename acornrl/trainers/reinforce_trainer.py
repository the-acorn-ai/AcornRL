from transformers import Trainer
import torch.nn.functional as F


class ReinforceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards = inputs.pop("reward")
        outputs = model(**inputs, return_dict=True)

        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        labels = inputs.get("labels", None)
        mask = (labels != -100).float()

        log_probs = F.log_softmax(logits, dim=-1)
        gather_indices = labels.clone()
        gather_indices[gather_indices == -100] = 0

        gathered_log_probs = log_probs.gather(dim=-1, index=gather_indices.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = gathered_log_probs * mask
        sequence_log_probs = masked_log_probs.sum(dim=-1)

        # Policy gradient loss = - E[R * log_probs]
        pg_loss = -(rewards.float() * sequence_log_probs).mean()

        if return_outputs:
            outputs.loss = pg_loss
            return pg_loss, outputs
        else:
            return pg_loss