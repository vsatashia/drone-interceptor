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

    def __init__(self, field_of_view=90, init_quadrotor_state=np.zeros(12), init_hoop_radius=0.5, init_hoop_pose=np.zeros(6)):
        """
        Constructor for the HoopCam object
        """
        # Camera parameters
        self.fov = field_of_view # [degrees]

        # Hoop parameters
        self.prev_hoop_radius: float = 0.5
        self.prev_hoop_pose: np.ndarray = np.zeros(6)
        self.hoop_radius: float = init_hoop_radius
        self.hoop_pose: np.ndarray = init_hoop_pose
        
        # Quadrotor parameters
        self.quadrotor_state: np.ndarray = init_quadrotor_state

    def measurement(self, quadrotor_state, hoop_radius, hoop_pose, dt):
        """
        Simulate a camera measurement of the hoop's pose
        """

        # Update the quadrotor's state
        self.quadrotor_state = quadrotor_state

        # Relative velocity to find blur direction
        v_drone = quadrotor_state[6:9]
        v_hoop = (hoop_pose[:3] - self.prev_hoop_pose[:3]) / dt
        blur_direction = (v_hoop - v_drone) / np.linalg.norm(v_hoop - v_drone) if (np.linalg.norm(v_hoop - v_drone) > 0.001) else np.zeros(3)
        blur_strength = 10*np.linalg.norm(v_hoop - v_drone) * dt
        
        # Apply motion blur to the hoop
        estimated_pose = self.motion_blur(hoop_pose, blur_strength, blur_direction)
        self.prev_hoop_pose = hoop_pose
        self.prev_hoop_radius = hoop_radius
        self.hoop_pose = estimated_pose
        return self.hoop_pose

    def motion_blur(self, hoop_pose, blur_strength, blur_direction):
        """
        Simulate motion blur on moving hoop relative to the moving quadrotor
        
        Inputs:
            drone_state (6,): State of the drone, format: (x, y, z, roll, pitch, yaw, vx, vy, vz, v_roll, v_pitch, v_yaw)
            hoop_state (6,): State of the hoop, format: (x, y, z, roll, pitch, yaw)
            blur_strength (float): Strength of motion blur
            blur_direction (3,): Direction of motion blur in world coordinates
        
        Outputs:
            hoop_estimate (6,): Noisy motion-blurred estimate of hoop pose
        """

        blur_xyz = blur_direction
        blur_rpy = np.zeros(3)

        blur = np.concatenate((blur_xyz, blur_rpy))
        
        # Generate random noise based on normal distribution
        scale_factor = 0.05 # part that needs to be tuned for realistic noise
        noise = np.random.normal(scale= scale_factor * blur_strength) * blur
        
        # Add noise to the hoop position and orientation
        hoop_estimate = hoop_pose + noise
        
        return hoop_estimate


if __name__ == "__main__":
    def simulate_motion_and_measurements(quadrotor_trajectory, hoop_trajectory, dt):
        cam = HoopCam()
        
        hoop_poses = []
        
        for i in range(len(quadrotor_trajectory)):
            quad_state = quadrotor_trajectory[i]
            hoop_pose = hoop_trajectory[i]
            
            measured_pose = cam.measurement(quad_state, 0.5, hoop_pose, dt)

            hoop_poses.append(measured_pose)
        
        return np.array(hoop_poses)

    # Generate example trajectories
    num_steps = 1000
    dt = 1

    # Quadrotor trajectory
    quadrotor_trajectory = np.zeros((num_steps, 12))
    quadrotor_trajectory[:, 0] = 1 * np.cos(2 * np.linspace(-2*np.pi, 2*np.pi, num_steps))  # sine x motion
    quadrotor_trajectory[:, 6] = 1 * np.sin(2 * np.linspace(-2*np.pi, 2*np.pi, num_steps))  # cosine velocity in x

    # Hoop trajectory
    hoop_trajectory = np.zeros((num_steps, 6))
    hoop_trajectory[:, 0] = np.cos(10*np.linspace(0, 2*np.pi, num_steps))  # stationary
    hoop_trajectory[:, 1] = 10  # Constant y position

    # Simulate motion and measurements
    measured_hoop_poses = simulate_motion_and_measurements(quadrotor_trajectory, hoop_trajectory, dt)

    # Plot the results
    fig, axs = plt.subplots(2, 1)

    # Plot quadrotor trajectory
    axs[0].plot(quadrotor_trajectory[:, 0], label='Quadrotor x position')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Position')
    axs[0].legend()

    # Plot measured hoop x position over time
    axs[1].plot(measured_hoop_poses[:, 0], label='Measured Hoop x position')
    axs[1].plot(hoop_trajectory[:, 0], label='True Hoop x position', linestyle='--')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Position')
    axs[1].legend()

    plt.tight_layout()
    plt.show()