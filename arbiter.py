"""
Copyright (c) 2006-2007 Erin Catto http://www.gphysics.com

Permission to use, copy, modify, distribute and sell this software
and its documentation for any purpose is hereby granted without fee,
provided that the above copyright notice appear in all copies.
Erin Catto makes no representations about the suitability
of this software for any purpose.
It is provided "as is" without express or implied warranty.
"""
from copy import copy
from typing import Sequence

import torch

from body import Body
from collide import Contact, collide
from math_utils import cross
from settings import ACCUMULATE_IMPULSES, POSITION_CORRECTION, WARM_STARTING


class ArbiterKey:
    """Use as a key for a dictionary of Arbiters.
    Evaluates to same if contained bodies are the same.
    """
    def __init__(self, body_1: Body, body_2: Body) -> None:
        # define consistent ordering for attributes
        if id(body_1) < id(body_2):
            self.body_1 = body_1
            self.body_2 = body_2
        else:
            self.body_1 = body_2
            self.body_2 = body_1

    def __eq__(self, other: 'ArbiterKey') -> bool:
        return self.body_1 == other.body_1 and self.body_2 == other.body_2

    def __hash__(self) -> int:
        return hash((self.body_1, self.body_2))


class Arbiter:
    def __init__(self, body_1: Body, body_2: Body) -> None:
        if id(body_1) < id(body_2):
            self.body_1 = body_1
            self.body_2 = body_2
        else:
            self.body_1 = body_2
            self.body_2 = body_1

        self.contacts = collide(self.body_1, self.body_2)

        self.friction = torch.sqrt(self.body_1.friction * self.body_2.friction)

    @property
    def num_contacts(self):
        return len(self.contacts)

    def update(self, new_contacts: Sequence[Contact]) -> None:
        merged_contacts = []
        # TODO Can maybe optimize this matching/search process using a dict?
        for c_new in new_contacts:
            c_old = None
            for c_old in self.contacts:
                if c_new.edges == c_old.edges:
                    break

            if c_old is not None:
                c = c_new  # TODO Is it necessary to copy(c_new)?
                merged_contacts.append(c)
                if WARM_STARTING:
                    c.Pn = c_old.Pn
                    c.Pt = c_old.Pt
                    c.Pnb = c_old.Pnb
                else:
                    c.Pn = 0.0
                    c.Pt = 0.0
                    c.Pnb = 0.0
            else:
                merged_contacts.append(c_new)  # TODO Is it necessary to copy(c_new)?

        self.contacts = merged_contacts

    def pre_step(self, inv_dt: float) -> None:
        k_allowed_penetration = 0.01
        k_bias_factor = 0.2 if POSITION_CORRECTION else 0.0

        for c in self.contacts:
            r1 = c.position - self.body_1.position
            r2 = c.position - self.body_2.position

            # precompute normal mass, tangent mass, and bias
            rn1 = r1 @ c.normal
            rn2 = r2 @ c.normal
            k_normal = self.body_1.inv_mass + self.body_2.inv_mass
            k_normal += (self.body_1.inv_I * ((r1 @ r1) - rn1 * rn1)
                         + self.body_2.inv_I * ((r2 @ r2) - rn2 * rn2))
            c.mass_normal = 1.0 / k_normal

            tangent = cross(c.normal, 1.0)
            rt1 = r1 @ tangent
            rt2 = r2 @ tangent
            k_tangent = self.body_1.inv_mass + self.body_2.inv_mass
            k_tangent += (self.body_1.inv_I * ((r1 @ r1) - rt1 * rt1)
                          + self.body_2.inv_I * ((r2 @ r2) - rt2 * rt2))
            c.mass_tangent = 1.0 / k_tangent

            c.bias = -k_bias_factor * inv_dt * min(0.0, c.separation + k_allowed_penetration)

            if ACCUMULATE_IMPULSES:
                # apply normal + friction impulse
                P = c.Pn * c.normal + c.Pt * tangent

                self.body_1.velocity -= self.body_1.inv_mass * P
                self.body_1.angular_velocity -= self.body_1.inv_I * cross(r1, P)

                self.body_2.velocity += self.body_2.inv_mass * P
                self.body_2.angular_velocity += self.body_2.inv_I * cross(r2, P)

    def apply_impulse(self) -> None:
        b1 = self.body_1
        b2 = self.body_2

        for c in self.contacts:
            c.r1 = c.position - b1.position
            c.r2 = c.position - b2.position

            # relative velocity at contact
            dv = (b2.velocity + cross(b2.angular_velocity, c.r2)
                  - b1.velocity - cross(b1.angular_velocity, c.r1))

            # compute normal impulse
            vn = dv @ c.normal

            d_Pn = c.mass_normal * (-vn + c.bias)

            if ACCUMULATE_IMPULSES:
                # clamp the accumulated impulse
                Pn_0 = c.Pn
                c.Pn = max(Pn_0 + d_Pn, 0.0)
                d_Pn = c.Pn - Pn_0
            else:
                d_Pn = max(d_Pn, 0.0)

            # apply contact impulse
            Pn = d_Pn * c.normal

            b1.velocity -= b1.inv_mass * Pn
            b1.angular_velocity -= b1.inv_I * cross(c.r1, Pn)

            b2.velocity += b2.inv_mass * Pn
            b2.angular_velocity += b2.inv_I * cross(c.r2, Pn)

            # relative velocity at contact
            dv = (b2.velocity + cross(b2.angular_velocity, c.r2)
                  - b1.velocity - cross(b1.angular_velocity, c.r1))

            tangent = cross(c.normal, 1.0)
            vt = dv @ tangent
            d_Pt = c.mass_tangent * (-vt)

            if ACCUMULATE_IMPULSES:
                # compute friction impulse
                max_Pt = self.friction * c.Pn

                # clamp friction
                old_tangent_impulse = c.Pt
                c.Pt = torch.min(torch.max(old_tangent_impulse + d_Pt, -max_Pt), max_Pt)  # TODO Does this work for gradient propagation?
                d_Pt = c.Pt - old_tangent_impulse
            else:
                max_Pt = self.friction * d_Pn
                d_Pt = torch.min(torch.max(d_Pt, -max_Pt), max_Pt)  # TODO Does this work for gradient propagation?

            # apply contact impulses
            Pt = d_Pt * tangent

            b1.velocity -= b1.inv_mass * Pt
            b1.angular_velocity -= b1.inv_I * cross(c.r1, Pt)

            b2.velocity += b2.inv_mass * Pt
            b2.angular_velocity += b2.inv_I * cross(c.r2, Pt)


if __name__ == '__main__':
    b1 = Body(torch.tensor([1.0, 2.0]))
    b2 = Body(torch.tensor([3.0, 4.0]))
    a = Arbiter(b1, b2)
    ak1 = ArbiterKey(b1, b2)
    ak2 = ArbiterKey(b2, b1)
    d = {ak1: a}
    assert ak1 in d and ak2 in d and hash(ak1) == hash(ak2) and ak1 == ak2
