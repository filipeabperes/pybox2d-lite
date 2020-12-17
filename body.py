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

import torch


DTYPE = torch.float


class Body:
    def __init__(self,
                 width: Union[torch.Tensor, Sequence[float]] = torch.zeros(2, dtype=DTYPE),
                 mass: float = float('inf')) -> None:
        self.width = torch.as_tensor(width)
        self.mass = torch.as_tensor(mass)

        self._position = torch.zeros(2, dtype=self.width.dtype)
        self._rotation = torch.zeros(1, dtype=self.width.dtype)

        self._velocity = torch.zeros(2, dtype=self.width.dtype)
        self._angular_velocity = torch.zeros(1, dtype=self.width.dtype)

        self._force = torch.zeros(2, dtype=self.width.dtype)
        self._torque = torch.zeros(1, dtype=self.width.dtype)

        self._friction = torch.tensor(0.2, dtype=self.width.dtype)

        if self.mass < float('inf'):
            self.inv_mass = 1.0 / self.mass
            self.I = self.mass * (self.width @ self.width) / 12.0
            self.inv_I = 1.0 / self.I
        else:
            self.inv_mass = 0.0
            self.I = float('inf')
            self.inv_I = 0.0

    def add_force(self, f: torch.Tensor) -> None:
        self.force += f

    def rotation_matrix(self) -> torch.Tensor:
        c = torch.cos(self.rotation)
        s = torch.sin(self.rotation)
        return torch.tensor([[c, -s],
                             [s, c]], dtype=self.width.dtype)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = torch.as_tensor(position, dtype=self.width.dtype)

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        self._rotation = torch.as_tensor(rotation, dtype=self.width.dtype).view(1)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = torch.as_tensor(velocity, dtype=self.width.dtype)

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity):
        self._angular_velocity = torch.as_tensor(angular_velocity, dtype=self.width.dtype).view(1)

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, force):
        self._force = torch.as_tensor(force, dtype=self.width.dtype)

    @property
    def torque(self):
        return self._torque

    @torque.setter
    def torque(self, torque):
        self._torque = torch.as_tensor(torque, dtype=self.width.dtype).view(1)

    @property
    def friction(self):
        return self._friction

    @friction.setter
    def friction(self, friction):
        self._friction = torch.as_tensor(friction, dtype=self.width.dtype).view(1)
