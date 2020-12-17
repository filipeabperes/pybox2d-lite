from typing import Union

import torch


def cross(a: Union[torch.Tensor, float], b: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
    if isinstance(a, torch.Tensor) and a.shape[0] == 2:
        if isinstance(b, torch.Tensor) and b.shape[0] == 2:
            # a and b are 2D vectors
            return a[0] * b[1] - a[1] * b[0]
        else:
            # a is 2D vector, b is scalar
            return torch.tensor((b * a[1], -b * a[0]), dtype=a.dtype)
    elif isinstance(b, torch.Tensor) and b.shape[0] == 2:
        # b is 2D vector, a is scalar
        return torch.tensor((-a * b[1], a * b[0]), dtype=b.dtype)
    else:
        # a and b are scalars
        return a * b
