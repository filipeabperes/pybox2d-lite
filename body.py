"""
Copyright (c) 2006-2007 Erin Catto http://www.gphysics.com

Permission to use, copy, modify, distribute and sell this software
and its documentation for any purpose is hereby granted without fee,
provided that the above copyright notice appear in all copies.
Erin Catto makes no representations about the suitability
of this software for any purpose.
It is provided "as is" without express or implied warranty.
"""
from typing import Sequence, Union

import numpy as np


DTYPE = float


class Body:
    def __init__(self,
                 width: Union[np.ndarray, Sequence[float]] = np.zeros(2, dtype=DTYPE),
                 mass: float = float('inf')) -> None:
        self.width = np.asarray(width)
        self.mass = mass

        self.position = np.zeros(2, dtype=self.width.dtype)
        self.rotation = 0.0

        self.velocity = np.zeros(2, dtype=self.width.dtype)
        self.angular_velocity = 0.0

        self.force = np.zeros(2, dtype=self.width.dtype)
        self.torque = 0.0

        self.friction = 0.2

        if self.mass < float('inf'):
            self.inv_mass = 1.0 / self.mass
            self.I = self.mass * (self.width @ self.width) / 12.0
            self.inv_I = 1.0 / self.I
        else:
            self.inv_mass = 0.0
            self.I = float('inf')
            self.inv_I = 0.0

    def add_force(self, f: np.ndarray) -> None:
        self.force += f

    def rotation_matrix(self) -> np.ndarray:
        c = np.cos(self.rotation)
        s = np.sin(self.rotation)
        return np.array([[c, -s],
                         [s, c]], dtype=self.width.dtype)
