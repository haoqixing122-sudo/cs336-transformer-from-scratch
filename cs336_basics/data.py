import torch
import numpy as np
import numpy.typing as npt


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    max_idx = len(dataset) - context_length - 1
    if max_idx < 0:
        raise ValueError("dataset must contain at least context_length + 1 tokens")
    ix = torch.randint(0, max_idx + 1, (batch_size,)).tolist()
    x = np.stack([dataset[i:i + context_length] for i in ix])
    y = np.stack([dataset[i + 1:i + context_length + 1] for i in ix])
    return (torch.from_numpy(x).to(device=device, dtype=torch.long),
            torch.from_numpy(y).to(device=device, dtype=torch.long))
