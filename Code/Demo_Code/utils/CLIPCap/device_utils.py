import torch

# Device constants
CPU = torch.device('cpu')

def get_device(device_id: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


# Example: Use `CUDA` as the default device
CUDA = get_device(0)
