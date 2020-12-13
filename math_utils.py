import numpy as np
from typing import Union


def cross(a: Union[np.ndarray, float], b: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(a, np.ndarray) and a.shape[0] == 2:
        if isinstance(b, np.ndarray) and b.shape[0] == 2:
            # a and b are 2D vectors
            return np.cross(a, b)
        else:
            # a is 2D vector, b is scalar
            return np.array((b * a[1], -b * a[0]), dtype=a.dtype)
    elif isinstance(b, np.ndarray) and b.shape[0] == 2:
        # b is 2D vector, a is scalar
        return np.array((-a * b[1], a * b[0]), dtype=b.dtype)
    else:
        # a and b are scalars
        return a * b
