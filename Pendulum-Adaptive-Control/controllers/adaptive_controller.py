"""
Adaptive controller for the simple pendulum based on MRAC principle.
"""
import numpy as np

class AdaptiveController:
    def __init__(self, m=1.0, l=1.0, k_p=5.0, k_d=2.0, gamma=10.0):
        # physical parameters
        self.m = m
        self.l = l
        # PD baseline gains
        self.k_p = k_p
        self.k_d = k_d
        # adaptation gain
        self.gamma = gamma
        # parameter estimate for unknown disturbance coefficient
        self.theta_hat = 0.0

    def control(self, theta, theta_dot, theta_ref=0.0, theta_dot_ref=0.0):
        """
        Compute control torque and update parameter estimate.
        theta: current angle
        theta_dot: current angular velocity
        theta_ref: reference angle (default 0)
        theta_dot_ref: reference velocity
        """
        # tracking error
        e = theta - theta_ref
        edot = theta_dot - theta_dot_ref
        # regressor: gravity term for error coordinate
        phi = np.sin(theta - theta_ref)
        # MRAC control law: u = -k_p*e - k_d*edot - theta_hat*phi
        u = -self.k_p * e - self.k_d * edot - self.theta_hat * phi
        # update law: ˙theta_hat = gamma * phi * e
        self.theta_hat += self.gamma * phi * e * self.dt
        return u

    def set_dt(self, dt):
        """
        Set integration time step (used in adaptation update)
        """
        self.dt = dt