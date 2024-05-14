"""
Imports
"""
import numpy as np

from rotorpy.trajectories.polynomial_traj import Polynomial

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

        self.traj_start_time = 0

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

        pt1 = self.start
        v1 = np.array([0, 0, 0])
        yaw1 = 0

        if hasattr(self, 'traj'):
            flat_output = self.traj.update(t - self.traj_start_time)

            pt1 = flat_output['x']
            v1 = flat_output['x_dot']
            yaw1 = flat_output['yaw']

        # TODO update destination point
        pt2 = np.array([np.cos(2 * np.pi / 15 * t), np.sin(2 * np.pi / 15 * t), 1])
        v2 = np.array([0, 0, 0])

        self.traj = Polynomial(points=np.array([pt1, pt2]), v_avg=10)
        self.traj_start_time = t

        return self.traj.update(0)