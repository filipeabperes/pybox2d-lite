"""
Copyright (c) 2006-2007 Erin Catto http://www.gphysics.com

Permission to use, copy, modify, distribute and sell this software
and its documentation for any purpose is hereby granted without fee,
provided that the above copyright notice appear in all copies.
Erin Catto makes no representations about the suitability
of this software for any purpose.
It is provided "as is" without express or implied warranty.
"""


from typing import List, Dict, Union, Sequence

import torch

from arbiter import Arbiter, ArbiterKey
from body import Body
from joint import Joint


class World:
	def __init__(self, gravity:  Union[torch.Tensor, Sequence[float]], iterations: int) -> None:
		self.gravity = torch.as_tensor(gravity)
		self.iterations = iterations

		self.clear()

	def add_body(self, body: Body) -> None:
		self.bodies.append(body)

	def add_joint(self, joint: Joint) -> None:
		self.joints.append(joint)

	def clear(self) -> None:
		self.bodies: List[Body] = []
		self.joints: List[Joint] = []
		self.arbiters: Dict[ArbiterKey, Arbiter] = {}

	def broadphase(self) -> None:
		# O(n^2) broad-phase
		for i, bi in enumerate(self.bodies):
			for j, bj in enumerate(self.bodies[i+1:]):
				if bi.inv_mass == 0.0 and bj.inv_mass == 0.0:
					continue

				new_arb = Arbiter(bi, bj)
				key = ArbiterKey(bi, bj)

				if new_arb.num_contacts > 0:
					if key not in self.arbiters:
						self.arbiters[key] = new_arb
					else:
						self.arbiters[key].update(new_arb.contacts)
				elif key in self.arbiters:
					self.arbiters.pop(key)

	def step(self, dt: float) -> None:
		inv_dt = 1.0 / dt if dt > 0 else 0.0

		# determine overlapping bodies and update contact points
		self.broadphase()

		# integrate forces
		for b in self.bodies:
			if b.inv_mass == 0:
				continue

			b.velocity += dt * (self.gravity + b.inv_mass * b.force)
			b.angular_velocity += dt * b.inv_I * b.torque

		# perform pre-steps
		for arb in self.arbiters.values():
			arb.pre_step(inv_dt)

		for joint in self.joints:
			joint.pre_step(inv_dt)

		# perform iterations
		for _ in range(self.iterations):
			for arb in self.arbiters.values():
				arb.apply_impulse()

			for joint in self.joints:
				joint.apply_impulse()

		# integrate velocities
		for b in self.bodies:
			b.position += dt * b.velocity
			b.rotation += dt * b.angular_velocity

			b.force = torch.zeros(2, dtype=b.force.dtype)
			b.torque = 0.0
