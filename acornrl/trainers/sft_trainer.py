from transformers import Trainer
import torch.nn.functional as F
from .utils import compute_log_likelihood
import torch

class SFTTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        with torch.no_grad():
            # model.ref_model.eval()
            ref_model_outputs = model.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            ref_logprob = compute_log_likelihood(ref_model_outputs.logits, inputs['labels']) #[batch_size]

        logprob = compute_log_likelihood(model_outputs.logits, inputs['labels'])

        # for MC kl
        kl_divergence = logprob.exp() * (logprob - ref_logprob)

        loss = - logprob + self.args.lm_kl_coeff * kl_divergence        
        
        total_loss = (loss * inputs['weights']).mean() # [batch_size]

        return (total_loss, model_outputs) if return_outputs else total_loss