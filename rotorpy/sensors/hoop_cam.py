import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from collections import deque

class HoopCam:
    """
    Simulate a camera that is mounted on a quadrotor and is looking at a hoop.
    This class aims to provide a realistic (noisy) idea of how a camera + CV algorithm might estimate the
    pose of a dynamic object (the hoop) in the camera's FOV while moving on a quadrotor
    """

    def __init__(self, field_of_view=90, init_quadrotor_state=np.zeros(12), init_hoop_radius=0.5, init_hoop_pose=np.eye(4)):
        """
        Constructor for the HoopCam object
        """
        # Camera parameters
        self.fov = field_of_view # [degrees]

        # Hoop parameters
        self.prev_hoop_radius = 0.5
        self.prev_hoop_pose = np.eye(4)
        self.hoop_radius = init_hoop_radius
        self.hoop_pose = init_hoop_pose
        
        self.quadrotor_state = init_quadrotor_state

    def measurement(self, quadrotor_state, hoop_radius, hoop_pose):
        """
        Simulate a camera measurement of the hoop's pose
        """
        # Update the hoop's pose and radius
        self.prev_hoop_radius = self.hoop_radius
        self.prev_hoop_pose = self.hoop_pose
        self.hoop_radius = hoop_radius
        self.hoop_pose = hoop_pose

        # Update the quadrotor's state
        self.quadrotor_state = quadrotor_state

        # Simulate the camera measurement
        return self.hoop_pose


