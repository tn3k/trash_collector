#!/usr/bin/env python

# Author: Kent Koehler
# Date: 5/21/2023

# Referenced PD controller seen in CS81

class PD():
    """
    Simple PD class to separate control from node
    """
    def __init__(self, kp, kd):
        """
        Initialize PD controller with gains of kp and kd

        :param kp: The proportional gain constant
        :param kd: The derivative gain constant
        """

        self._p = kp
        self._d = kd
        self._err_prev = None
        self._last_time = None

    def step(self, err, time):
        """
        Called on each sensor update to return an angular velocity command that
        will turn the robot away from or toward the right wall.

        :param err: The current error of the robot
        :param time: Current time
        """
        # default for starting error
        u = 0
        if self._err_prev is not None and self._last_time is not None:
            dt = time - self._last_time
            # if previous error set then calculate velocity command with it
            if dt == 0:
                u = self._p * err
            else:
                u = self._p * err + self._d * (err - self._err_prev) / dt
        # set new previous error and return actuation command
        self._err_prev = self._p * err
        self._last_time = time
        return u