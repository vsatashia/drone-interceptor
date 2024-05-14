"""
Imports
"""
import numpy as np

from rotorpy.trajectories.minsnap import MinSnap

class HoopTraj(object):
    """
    The trajectory is implemented as a class. There are two required methods for each trajectory class: __init__() and update().
    The __init__() is required for instantiating the class with appropriate parameterizations. For example, if you are doing 
    a circular trajectory you might want to specify the radius of the circle. 
    The update() method is called at each iteration of the simulator. The only input to update is time t. The output of update()
    should be the desired flat outputs in a dictionary, as specified below. 
    """
    def __init__(self):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.
        """

        # TODO add starting point
        self.start = np.array([0, 0, 0])

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """

        if not hasattr(self, 'ms'):
            pt1 = self.start
            v1 = np.array([0, 0, 0])
            yaw1 = 0
        else:
            flat_output = self.ms.update(t - self.ms_start_time)

            pt1 = flat_output['x']
            v1 = flat_output['x_dot']
            yaw1 = flat_output['yaw']

        # TODO update destination point
        pt2 = np.array([2, -1, 1.5])
        v2 = np.array([0, 0, 0])

        self.ms = MinSnap(points=np.array([pt1, pt2]), yaw_angles=np.array([yaw1, 0]), v_start=v1, v_end=v2, v_avg=3, v_max=5)
        self.ms_start_time = t

        return self.ms.update(0)