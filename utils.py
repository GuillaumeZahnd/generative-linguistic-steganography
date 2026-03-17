import torch


def reset_seed(seed: int) -> None:
    """Reset all random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
