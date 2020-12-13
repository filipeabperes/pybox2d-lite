"""
Copyright (c) 2006-2007 Erin Catto http://www.gphysics.com

Permission to use, copy, modify, distribute and sell this software
and its documentation for any purpose is hereby granted without fee,
provided that the above copyright notice appear in all copies.
Erin Catto makes no representations about the suitability
of this software for any purpose.
It is provided "as is" without express or implied warranty.
"""

import numpy as np
from enum import Enum
from typing import Optional, Sequence, List, Union, Tuple

from body import Body


"""
Box vertex and edge numbering:
       ^ y
       |
       e1
  v2 ------ v1
   |        |
e2 |        | e4  --> x
   |        |
  v3 ------ v4
       e3
"""


class Contact:
    def __init__(self, Pn: float = 0.0, Pt: float = 0.0, Pnb: float = 0.0) -> None:
        self.Pn = Pn
        self.Pt = Pt
        self.Pnb = Pnb

        self.position: np.ndarray = None
        self.normal: np.ndarray = None
        self.r1: np.ndarray = None
        self.r2: np.ndarray = None

        self.separation: float = None
        self.mass_normal: float = None
        self.mass_tangent: float = None
        self.bias: float = None

        self.edges: Edges = None


class Axis(Enum):
    FACE_A_X = 1
    FACE_A_Y = 2
    FACE_B_X = 3
    FACE_B_Y = 4


class EdgeNumbers(Enum):
    NO_EDGE = 0
    EDGE_1 = 1
    EDGE_2 = 2
    EDGE_3 = 3
    EDGE_4 = 4


class Edges:
    def __init__(self,
                 in_edge_1: EdgeNumbers = EdgeNumbers.NO_EDGE,
                 out_edge_1: EdgeNumbers = EdgeNumbers.NO_EDGE,
                 in_edge_2: EdgeNumbers = EdgeNumbers.NO_EDGE,
                 out_edge_2: EdgeNumbers = EdgeNumbers.NO_EDGE) -> None:
        self.in_edge_1 = in_edge_1
        self.out_edge_1 = out_edge_1
        self.in_edge_2 = in_edge_2
        self.out_edge_2 = out_edge_2

    def __eq__(self, other: 'Edges') -> bool:
        return (self.in_edge_1 == other.in_edge_1 and
                self.out_edge_1 == other.out_edge_1 and
                self.in_edge_2 == other.in_edge_2 and
                self.out_edge_2 == other.out_edge_2)

    def __hash__(self) -> int:
        return hash((self.in_edge_1, self.out_edge_1, self.in_edge_2, self.out_edge_2))

    def flip(self) -> None:
        self.in_edge_1, self.in_edge_2 = self.in_edge_2, self.in_edge_1
        self.out_edge_1, self.out_edge_2 = self.out_edge_2, self.out_edge_1


class ClipVertex:
    def __init__(self, v: Optional[np.ndarray] = None):
        self.edges = Edges()
        self.v = v


def clip_segment_to_line(v_in: Sequence[ClipVertex], normal: np.ndarray,
                         offset: float, clip_edge: EdgeNumbers) -> List[ClipVertex]:
    v_out = []

    distance_0 = normal @ v_in[0].v - offset
    distance_1 = normal @ v_in[1].v - offset

    if distance_0 <= 0:
        v_out.append(v_in[0])

    if distance_1 <= 0:
        v_out.append(v_in[1])

    if distance_0 * distance_1 < 0:
        interp = distance_0 / (distance_0 - distance_1)
        new_v = ClipVertex(v=v_in[0].v + interp * (v_in[1].v - v_in[0].v))

        if distance_0 > 0:
            new_v.edges = v_in[0].edges
            new_v.edges.in_edge_1 = clip_edge
            new_v.edges.in_edge_2 = EdgeNumbers.NO_EDGE
        else:
            new_v.edges = v_in[1].edges
            new_v.edges.out_edge_1 = clip_edge
            new_v.edges.out_edge_2 = EdgeNumbers.NO_EDGE

        v_out.append(new_v)

    assert len(v_out) <= 2
    return v_out


def compute_incident_edge(h: np.ndarray, pos: np.ndarray,
                          rot: np.ndarray, normal: np.ndarray) -> Tuple[ClipVertex, ClipVertex]:
    c = (ClipVertex(), ClipVertex())
    # The normal is from the reference box.
    #  Convert it to the incident box's frame and flip sign.
    rot_t = rot.transpose()
    n = -(rot_t @ normal)
    n_abs = np.abs(n)

    if n_abs[0] > n_abs[1]:
        if np.sign(n[0]) > 0:
            c[0].v = np.array([h[0], -h[1]], dtype=h.dtype)
            c[0].edges.in_edge_2 = EdgeNumbers.EDGE_3
            c[0].edges.out_edge_2 = EdgeNumbers.EDGE_4

            c[1].v = h.copy()
            c[1].edges.in_edge_2 = EdgeNumbers.EDGE_4
            c[1].edges.out_edge_2 = EdgeNumbers.EDGE_1
        else:
            c[0].v = np.array([-h[0], h[1]], dtype=h.dtype)
            c[0].edges.in_edge_2 = EdgeNumbers.EDGE_1
            c[0].edges.out_edge_2 = EdgeNumbers.EDGE_2

            c[1].v = -h
            c[1].edges.in_edge_2 = EdgeNumbers.EDGE_2
            c[1].edges.out_edge_2 = EdgeNumbers.EDGE_3
    else:
        if np.sign(n[1]) > 0:
            c[0].v = h.copy()
            c[0].edges.in_edge_2 = EdgeNumbers.EDGE_4
            c[0].edges.out_edge_2 = EdgeNumbers.EDGE_1

            c[1].v = np.array([-h[0], h[1]], dtype=h.dtype)
            c[1].edges.in_edge_2 = EdgeNumbers.EDGE_1
            c[1].edges.out_edge_2 = EdgeNumbers.EDGE_2
        else:
            c[0].v = -h
            c[0].edges.in_edge_2 = EdgeNumbers.EDGE_2
            c[0].edges.out_edge_2 = EdgeNumbers.EDGE_3

            c[1].v = np.array([h[0], -h[1]], dtype=h.dtype)
            c[1].edges.in_edge_2 = EdgeNumbers.EDGE_3
            c[1].edges.out_edge_2 = EdgeNumbers.EDGE_4

    c[0].v = pos + rot @ c[0].v
    c[1].v = pos + rot @ c[1].v
    return c


def collide(body_a: Body, body_b: Body) -> List[Contact]:
    # setup
    h_a = 0.5 * body_a.width
    h_b = 0.5 * body_b.width

    pos_a = body_a.position
    pos_b = body_b.position

    rot_a = body_a.rotation_matrix()
    rot_b = body_b.rotation_matrix()

    rot_a_t = rot_a.transpose()
    rot_b_t = rot_b.transpose()

    dp = pos_b - pos_a
    d_a = rot_a_t @ dp
    d_b = rot_b_t @ dp

    C = rot_a_t @ rot_b
    abs_C = np.abs(C)
    abs_C_t = abs_C.transpose()

    # box a faces
    face_a = np.abs(d_a) - h_a - abs_C @ h_b
    if face_a[0] > 0 or face_a[1] > 0:
        return []

    # box a faces
    face_b = np.abs(d_b) - abs_C_t @ h_a - h_b
    if face_b[0] > 0 or face_b[1] > 0:
        return []

    # find best axis
    axis = Axis.FACE_A_X
    separation = face_a[0]
    normal = rot_a[:, 0] if d_a[0] > 0 else -rot_a[:, 0]

    relative_tol = 0.95
    absolute_tol = 0.01

    if face_a[1] > relative_tol * separation + absolute_tol * h_a[1]:
        axis = Axis.FACE_A_Y
        separation = face_a[1]
        normal = rot_a[:, 1] if d_a[1] > 0 else -rot_a[:, 1]


    if face_b[0] > relative_tol * separation + absolute_tol * h_b[0]:
        axis = Axis.FACE_B_X
        separation = face_b[0]
        normal = rot_b[:, 0] if d_b[0] > 0 else -rot_b[:, 0]

    if face_b[1] > relative_tol * separation + absolute_tol * h_b[1]:
        axis = Axis.FACE_B_Y
        separation = face_b[1]
        normal = rot_b[:, 1] if d_b[1] > 0 else -rot_b[:, 1]

    # setup clipping plane data based on the separating axis
    # compute the clipping lines and the line segment to be clipped
    if axis == Axis.FACE_A_X:
        front_normal = normal
        front = pos_a @ front_normal + h_a[0]
        side_normal = rot_a[:, 1]
        side = pos_a @ side_normal
        neg_side = -side + h_a[1]
        pos_side = side + h_a[1]
        neg_edge = EdgeNumbers.EDGE_3
        pos_edge = EdgeNumbers.EDGE_1
        incident_edge = compute_incident_edge(h_b, pos_b, rot_b, front_normal)

    elif axis == Axis.FACE_A_Y:
        front_normal = normal
        front = pos_a @ front_normal + h_a[1]
        side_normal = rot_a[:, 0]
        side = pos_a @ side_normal
        neg_side = -side + h_a[0]
        pos_side = side + h_a[0]
        neg_edge = EdgeNumbers.EDGE_2
        pos_edge = EdgeNumbers.EDGE_4
        incident_edge = compute_incident_edge(h_b, pos_b, rot_b, front_normal)

    elif axis == Axis.FACE_B_X:
        front_normal = -normal
        front = pos_b @ front_normal + h_b[0]
        side_normal = rot_b[:, 1]
        side = pos_b @ side_normal
        neg_side = -side + h_b[1]
        pos_side = side + h_b[1]
        neg_edge = EdgeNumbers.EDGE_3
        pos_edge = EdgeNumbers.EDGE_1
        incident_edge = compute_incident_edge(h_a, pos_a, rot_a, front_normal)

    elif axis == Axis.FACE_B_Y:
        front_normal = -normal
        front = pos_b @ front_normal + h_b[1]
        side_normal = rot_b[:, 0]
        side = pos_b @ side_normal
        neg_side = -side + h_b[0]
        pos_side = side + h_b[0]
        neg_edge = EdgeNumbers.EDGE_2
        pos_edge = EdgeNumbers.EDGE_4
        incident_edge = compute_incident_edge(h_a, pos_a, rot_a, front_normal)

    else:
        raise ValueError(f'Axis valus set incorrectly ({axis}).')

    # clip other face with 5 box planes (1 face plane, 4 edge planes)
    #  clip to box side 1
    clip_points_1 = clip_segment_to_line(incident_edge, -side_normal, neg_side, neg_edge)
    if len(clip_points_1) < 2:
        return []

    #  clip to negative box side 1
    clip_points_2 = clip_segment_to_line(clip_points_1, side_normal, pos_side, pos_edge)
    if len(clip_points_2) < 2:
        return []

    # Now clipPoints2 contains the clipping points.
    #  Due to roundoff, it is possible that clipping removes all points.
    contacts = []
    for cp in clip_points_2:
        separation = front_normal @ cp.v - front
        if separation <= 0:
            new_contact = Contact()
            new_contact.separation = separation
            new_contact.normal = normal
            # slide contact point onto reference face (easy to cull)
            new_contact.position = cp.v - separation * front_normal
            new_contact.edges = cp.edges
            if axis == Axis.FACE_B_X or axis == Axis.FACE_B_Y:
                new_contact.edges.flip()
            contacts.append(new_contact)

    return contacts
