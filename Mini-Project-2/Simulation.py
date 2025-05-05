import numpy as np
import time

class Simulation:
    def __init__(self, dt, desired_path, wheel_radius, wheel_width,
                 m, I, dB=0.0, # Dynamic parameters: mass, inertia, disturbance bound
                 initial_pose=np.array([0.0, 0.0, 0.0]),
                 initial_velocity=np.array([0.0, 0.0])): # v1, omega
        """
        Initialize the simulation with dynamic model.
        :param dt: Time step duration.
        :param desired_path: A numpy array of shape (N, 2) representing the desired path.
        :param wheel_radius: Radius of the wheels (r).
        :param wheel_width: Distance between the wheels (W).
        :param m: Mass of the robot.
        :param I: Moment of inertia of the robot around center.
        :param dB: Upper bound for disturbance torque magnitude (optional).
        :param initial_pose: Initial pose (x, y, theta).
        :param initial_velocity: Initial velocity (v1, omega).
        """
        self.desired_path = desired_path
        self.wheel_radius = wheel_radius # r
        self.wheel_width = wheel_width   # W
        self.dt = dt

        # Dynamic Parameters
        self.m = m
        self.I = I
        self.dB = dB # Max disturbance torque (applied component-wise to v_dot eq)
        # Check for zero mass or inertia
        if m <= 0 or I <= 0:
             raise ValueError("Mass and Inertia must be positive.")
        self.M_inv = np.linalg.inv(np.diag([m, I])) # Inverse of Mass Matrix M2
        # Input Matrix B2, mapping tau=[tau_r, tau_l]^T to accelerations
        if wheel_radius <= 0:
             raise ValueError("Wheel radius must be positive.")
        self.B2 = (1.0 / self.wheel_radius) * np.array([
            [1.0, 1.0],
            [self.wheel_width / 2.0, -self.wheel_width / 2.0]
        ])

        # State variables
        self.x, self.y, self.theta = initial_pose
        self.v1, self.omega = initial_velocity # Forward and angular velocity

        # History Storage (Initialize with initial state/zeros consistent with time=0)
        self.actual_path = np.array([[self.x, self.y, self.theta]])  # Store pose (t=0)
        self.time_stamps = [0.0]
        self.torques_cmd = [(0.0, 0.0)] # Command applied *during* interval ending at t=0 (None) -> Use placeholder? Length N+1
        self.robot_velocities_actual = [initial_velocity.tolist()] # (v1, omega) at t=0
        self.robot_accelerations_actual = [(0.0, 0.0)] # Accel calculated during step k, affects state k+1. Start with 0. Length N+1
        self.disturbances_actual = [(0.0, 0.0)] # Disturbance during step k. Start with 0. Length N+1

    def _apply_disturbance(self):
        """ Generate some random disturbance torque vector """
        disturbance_v1 = np.random.uniform(-self.dB, self.dB) if self.dB > 0 else 0.0
        disturbance_omega = np.random.uniform(-self.dB, self.dB) if self.dB > 0 else 0.0
        tau_d_effective = np.array([disturbance_v1, disturbance_omega])
        return tau_d_effective


    def execute_cmd(self, tau_left, tau_right):
        """
        Update the robot's state based on the given wheel TORQUE commands.
        Simplified: M2*v_dot = B2*tau - tau_d_effective
        :param tau_left: Torque command for the left wheel (Nm).
        :param tau_right: Torque command for the right wheel (Nm).
        """
        # Store command *before* using it for the step k -> k+1 interval
        self.torques_cmd.append((tau_left, tau_right))

        # Input torque vector (tau_r, tau_l)
        tau_cmd = np.array([tau_right, tau_left])

        # Calculate actual disturbance for this step (applied during k -> k+1)
        tau_d_actual = self._apply_disturbance()
        self.disturbances_actual.append(tuple(tau_d_actual))

        # Calculate accelerations using the dynamic model
        v_dot = self.M_inv @ (self.B2 @ tau_cmd - tau_d_actual)
        v1_dot, omega_dot = v_dot
        self.robot_accelerations_actual.append([v1_dot, omega_dot])

        # Update velocities using Euler integration (state at k+1)
        self.v1 += v1_dot * self.dt
        self.omega += omega_dot * self.dt
        self.robot_velocities_actual.append([self.v1, self.omega])

        # Update the robot's position and orientation using kinematic model with *updated* velocities
        delta_x = self.v1 * np.cos(self.theta) * self.dt
        delta_y = self.v1 * np.sin(self.theta) * self.dt
        delta_theta = self.omega * self.dt

        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta

        # Normalize theta
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # Store updated pose (state at k+1)
        current_pose = np.array([[self.x, self.y, self.theta]])
        self.actual_path = np.append(self.actual_path, current_pose, axis=0)

        # Update time (time at k+1)
        self.time_stamps.append(self.time_stamps[-1] + self.dt)


    def get_robot_state(self):
        """ Get the current full state of the robot. """
        return self.x, self.y, self.theta, self.v1, self.omega

    def get_simulation_data(self):
        """ Returns all collected data, ensuring consistent lengths. """
        # All lists now have length N_steps + 1
        return {
            "time": np.array(self.time_stamps),
            "actual_path": self.actual_path, # x, y, theta
            "desired_path": self.desired_path, # x, y
            "torques_cmd": np.array(self.torques_cmd), # tau_l, tau_r (includes initial placeholder)
            "robot_vels": np.array(self.robot_velocities_actual), # v1, omega
            "robot_accels": np.array(self.robot_accelerations_actual), # v1_dot, omega_dot
            "disturbances": np.array(self.disturbances_actual) # Applied disturbance torque
        }