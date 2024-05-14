import numpy as np
from scipy.spatial.transform import Rotation
import copy
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

class HoopEstimator:
    """
    HoopEstimator
        Estimates the position of the hoop. 

        State space: 
                X = [x,y,z,xdot,ydot,zdot]
        Measurement space: 
                Y = [x,y,z]
    """

    def __init__(self,
                 Q=np.diag(np.concatenate([0.05*np.ones(3),0.07*np.ones(3)])),
                 R=np.diag(0.1*np.ones(3)),
                 xhat0=np.ones(6), 
                 P0=1*np.eye(6),
                 dt=1/100,
                 alpha=0.1,
                 beta=2.0,
                 kappa=-1):

        """
        Inputs:
            quad_params, dict with keys specified in quadrotor_params.py
            Q, the process noise covariance
            R, the measurement noise covariance
            x0, the initial filter state
            P0, the initial state covariance
            dt, the time between predictions
        """

        self.g = 9.81 # m/s^2

        # Filter parameters
        self.xhat = xhat0
        self.P = P0

        self.dt = dt

        self.N = self.xhat.shape[0]

        self.points = MerweScaledSigmaPoints(self.N, alpha=alpha, beta=beta, kappa=kappa)

        self.filter = UnscentedKalmanFilter(dim_x=self.N, dim_z=self.N, dt=dt, fx=self.f, hx=self.h, points=self.points)
        self.filter.x = xhat0
        self.filter.P = P0
        self.filter.R = R
        self.filter.Q = Q

        
    def step(self, camera_measurement):
        """
        The step command will update the filter based on the following. 
        Inputs:
            Camera measurement, noisy x,y,z position of the hoop.

        Outputs:
            A dictionary with the following keys: 
                filter_state, containing the current filter estimate.
                covariance, containing the current covariance matrix.
        """

        self.filter.predict()
        self.filter.update(camera_measurement)

        return {'filter_state': self.filter.x, 'covariance': self.filter.P}

    def f(self, xk, dt):
        """
        Dynamics model
        """
        xk[0:3] = xk[0:3] + xk[3:6]*dt
        return xk

    def h(self, xk):
        """
        Measurement model
        """
        h = xk[0:3]
        return h
