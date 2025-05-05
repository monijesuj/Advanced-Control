"""
Lyapunov (PD) controller for the pendulum (no parameter adaptation).
"""
import numpy as np

class LyapunovController:
    def __init__(self, m=1.0, l=1.0, k_p=15.0, k_d=5.0):
        # physical parameters
        self.m = m
        self.l = l
        # PD gains
        self.k_p = k_p
        self.k_d = k_d

    def control(self, theta, theta_dot, theta_ref=0.0, theta_dot_ref=0.0):
        """
        PD control law: u = -k_p*e - k_d*edot
        where e = theta - theta_ref, edot = theta_dot - theta_dot_ref
        """
        e = theta - theta_ref
        edot = theta_dot - theta_dot_ref
        u = -self.k_p * e - self.k_d * edot
        return u
