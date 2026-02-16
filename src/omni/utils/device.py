
import torch
from functools import lru_cache

@lru_cache(maxsize=1)
def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return torch.device(device)

if __name__ == "__main__":
    import timeit
    
    n = 100_000
    
    t = timeit.timeit(lambda: get_device(), number=n)
    print(f"Device: {repr(get_device())}, Time taken for {n:,} calls: {t:.4f} seconds")