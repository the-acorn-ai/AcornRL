import torch

# Constants
IGNORE_INDEX = -100

def compute_log_likelihood(logits, labels):
    """
    Compute the log likelihood for a batch of sequences.

    Args:
        logits: Model logits of shape (batch_size, seq_length, vocab_size)
        labels: Target labels of shape (batch_size, seq_length)

    Returns:
        Log likelihood for each sequence in the batch
    """
    batch_size, seq_length, vocab_size = logits.shape

    # Shift predictions and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels).reshape(batch_size, -1)

    # Compute average loss ignoring padding tokens
    ignore_mask = labels != IGNORE_INDEX
    avg_loss = loss.sum(dim=-1) / ignore_mask.sum(dim=-1).clamp(min=1)

    # Return negative loss as log likelihood
    return -avg_loss