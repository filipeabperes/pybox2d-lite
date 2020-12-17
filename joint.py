"""
Copyright (c) 2006-2007 Erin Catto http://www.gphysics.com

Permission to use, copy, modify, distribute and sell this software
and its documentation for any purpose is hereby granted without fee,
provided that the above copyright notice appear in all copies.
Erin Catto makes no representations about the suitability
of this software for any purpose.
It is provided "as is" without express or implied warranty.
"""

from typing import Optional

import torch

from body import Body
from math_utils import cross
from settings import WARM_STARTING, POSITION_CORRECTION


class Joint:
    def __init__(self,
                 body_1: Optional[Body] = None,
                 body_2: Optional[Body] = None,
                 anchor: Optional[torch.Tensor] = None) -> None:
        self.M = None
        self.P = None

        self.r1 = None
        self.r2 = None
        self.bias = None

        self.softness = 0.0
        self.bias_factor = 0.2

        self.body_1 = body_1
        self.body_2 = body_2
        if body_1 is not None and body_2 is not None and anchor is not None:
            rot_1 = body_1.rotation_matrix()
            rot_2 = body_2.rotation_matrix()
            self.local_anchor_1 = rot_1.t() @ (anchor - body_1.position)
            self.local_anchor_2 = rot_2.t() @ (anchor - body_2.position)

            self.P = torch.zeros(2, dtype=body_1.width.dtype)

    def pre_step(self, inv_dt: float) -> None:
        rot_1 = self.body_1.rotation_matrix()
        rot_2 = self.body_2.rotation_matrix()

        r1 = self.r1 = rot_1 @ self.local_anchor_1
        r2 = self.r2 = rot_2 @ self.local_anchor_2

        K1 = torch.tensor([[self.body_1.inv_mass + self.body_2.inv_mass, 0.0],
                           [0.0, self.body_1.inv_mass + self.body_2.inv_mass]])
        K2 = torch.tensor([[self.body_1.inv_I * r1[1] * r1[1], -self.body_1.inv_I * r1[0] * r1[1]],
                           [-self.body_1.inv_I * r1[0] * r1[1], self.body_1.inv_I * r1[0] * r1[0]]])
        K3 = torch.tensor([[self.body_2.inv_I * r2[1] * r2[1], -self.body_2.inv_I * r2[0] * r2[1]],
                           [-self.body_2.inv_I * r2[0] * r2[1], self.body_2.inv_I * r2[0] * r2[0]]])
        K = K1 + K2 + K3 + torch.eye(2) * self.softness
        self.M = torch.inverse(K)

        p1 = self.body_1.position + r1
        p2 = self.body_2.position + r2
        dp = p2 - p1

        if POSITION_CORRECTION:
            self.bias = -self.bias_factor * inv_dt * dp
        else:
            self.bias = torch.zeros(2, dtype=self.body_1.width.dtype)

        if WARM_STARTING:
            self.body_1.velocity -= self.body_1.inv_mass * self.P
            self.body_1.angular_velocity -= self.body_1.inv_I * cross(r1, self.P)
            self.body_2.velocity += self.body_2.inv_mass * self.P
            self.body_2.angular_velocity += self.body_2.inv_I * cross(r2, self.P)
        else:
            self.P = torch.zeros(2, dtype=self.body_1.width.dtype)

    def apply_impulse(self) -> None:
        dv = (self.body_2.velocity
              + cross(self.body_2.angular_velocity, self.r2)
              - cross(self.body_1.angular_velocity, self.r1))

        impulse = self.M @ (self.bias - dv - self.softness * self.P)

        self.body_1.velocity -= self.body_1.inv_mass * impulse
        self.body_1.angular_velocity -= self.body_1.inv_I * cross(self.r1, impulse)
        self.body_2.velocity += self.body_2.inv_mass * impulse
        self.body_2.angular_velocity += self.body_2.inv_I * cross(self.r2, impulse)

        self.P += impulse
