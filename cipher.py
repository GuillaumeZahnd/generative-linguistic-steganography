import torch

def stegano_encoding(
    top_probabilities: torch.Tensor,
    top_indices: torch.Tensor,
    payload_bit: int,
) -> tuple[int, int]:
    """
    Encode one bit per token using cumulative probabilities.

    Args:
        top_probabilities: Probability distribution, of shape (batch, top_k).
        top_indices: Vocabulary indices, of shape (batch, top_k).
        payload_bit: Bit to encode (either 0 or 1).

    Returns:
        Vocabulary index of selected token.
        Index in of the selected token in the top_k shortlist.
    """
    # Remove batch dimension if present
    if top_probabilities.dim() > 1:
        top_probabilities = top_probabilities.squeeze(0)
    if top_indices.dim() > 1:
        top_indices = top_indices.squeeze(0)

    device = top_probabilities.device

    # Cast to float32 for numerical stability
    probabilities_stable = top_probabilities.to(torch.float32)

    # Ensure sum is exactly 1.0
    probabilities_stable = probabilities_stable / probabilities_stable.sum()

    # Compute cumulative probabilities using the numerically stable probability values
    cumulative_probabilities = torch.cumsum(probabilities_stable, dim=-1)

    # Get the START of each token's probability range
    # Token 0 starts at 0, token i starts at cumulative_probabilities[i-1]
    starts = torch.cat([torch.tensor([0.0], device=device), cumulative_probabilities[:-1]])

    # Split: bit 0 if token starts in [0, 0.5), bit 1 if starts in [0.5, 1.0]
    if payload_bit == 0:
        mask = starts < 0.5
    else:
        mask = starts >= 0.5

    valid_indices = torch.where(mask)[0]

    if len(valid_indices) == 0:
        # Fallback: pick token whose START is closest to midpoint
        dists = torch.abs(starts - 0.5)
        selected_index = torch.argmin(dists).item()
    else:
        # Pick first valid token (closest to midpoint boundary)
        selected_index = valid_indices[0].item()

    selected_token = top_indices[selected_index].item()

    return selected_token, selected_index


def stegano_decoding(
    top_probabilities: torch.Tensor,
    top_indices: torch.Tensor,
    token_vocabulary_index: int
) -> int:
    """
    Decode one bit from a selected token index.

    Args:
        top_probabilities: Probability distribution, of shape (batch, top_k).
        top_indices: Vocabulary indices, of shape (batch, top_k).
        token_vocabulary_index: Actual vocabulary index of the token that was selected by the encoder.

    Returns:
        Payload bit (0 or 1).

    Raises:
        ValueError: If token_vocabulary_index is not in found among top_indices.
    """
    # Remove batch dimension if present
    if top_probabilities.dim() > 1:
        top_probabilities = top_probabilities.squeeze(0)
    if top_indices.dim() > 1:
        top_indices = top_indices.squeeze(0)

    # Find the index of token_vocabulary_index in top_indices
    token_position = torch.where(top_indices == token_vocabulary_index)[0]

    if len(token_position) == 0:
        raise ValueError(f"Token index {token_vocabulary_index} not found in top_k tokens.")

    token_topk_index = token_position[0].item()

    # Cast to float32 for numerical stability
    probabilities_stable = top_probabilities.to(torch.float32)

    # Ensure sum is exactly 1.0
    probabilities_stable = probabilities_stable / probabilities_stable.sum()

    # Compute cumulative probabilities using the numerically stable probability values
    cumulative_probabilities = torch.cumsum(probabilities_stable, dim=-1)

    # Get the start of the selected token's probability range
    if token_topk_index == 0:
        start = 0.0
    else:
        start = cumulative_probabilities[token_topk_index - 1].item()

    # Decode: if start < 0.5 then bit=0, otherwise bit=1
    payload_bit = 1 if start >= 0.5 else 0

    return payload_bit
