"""Feature functions for a basic xG model.

:author: Jesse Davis and Pieter Robberechts
:copyright: Copyright 2023 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import warnings

import numpy as np

_goal_x = 105.0
_goal_y = 34.0
_goal_width = 7.3


def dist_to_center_goal(x_pos, y_pos):
    """Compute the distance to the middle of the goal."""
    dx = _goal_x - x_pos
    dy = _goal_y - y_pos
    return np.sqrt(dx ** 2 + dy ** 2)


def angle_to_goal(x_pos, y_pos):
    """Compute the angle formed between the shot location and the two goal posts."""
    dx = _goal_x - x_pos
    dy = _goal_y - y_pos
    val = (_goal_width * dx) / (
        dx**2 + dy**2 - (_goal_width / 2) ** 2
    )

    angle = np.arctan(val)
    if angle < 0:
        angle += np.pi
    if x_pos > _goal_x:
        angle = 0
    if np.abs(x_pos - _goal_x) < 0.00001:
        warnings.warn("On the goal line")

    return angle
