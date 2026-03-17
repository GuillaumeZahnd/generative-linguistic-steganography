import torch
import torch.nn.functional as F


def get_top_probabilities(
    logits: torch.Tensor,
    top_k: int,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select the top-k tokens with highest probability.

    Args:
        logits: Raw output tensor from the language model, of shape (batch_size, vocabulary_size).
        top_k: Number of highest-probability tokens to keep.
        temperature: Scaling factor to control the sharpness of the distribution (<1.0: peaked, >1.0: uniform).

    Returns:
        Re-normalized top-k probabilities, of size (batch_size, top_k).
        Original vocabulary indices of the corresponding tokens, of size (batch_size, top_k).
    """

    if logits.dim() > 1:
        logits = logits.view(-1)

    assert top_k <= logits.size(-1), (
        f"top_k ({top_k}) cannot exceed vocabulary size ({logits.size(-1)})."
    )

    # Cast to float64 for numerical stability
    logits = logits.to(torch.float64)

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)

    # Compute probabilities
    top_probabilities = F.softmax(top_logits, dim=-1)
    
    # Ensure sum is exactly 1.0
    top_probabilities = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)    

    return top_probabilities.unsqueeze(0), top_indices.unsqueeze(0)
