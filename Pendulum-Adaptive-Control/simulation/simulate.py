# Add project root to PYTHONPATH so controllers module is found when script is run directly
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from controllers.adaptive_controller import AdaptiveController
import argparse

# physical constants
g = 9.81

# simulation parameters
dt = 0.001  # time step
T = 10.0    # total time
steps = int(T / dt)

time = np.linspace(0, T, steps)

# parse command-line arguments
parser = argparse.ArgumentParser(description='Pendulum adaptive control simulation')
parser.add_argument('--theta0', type=float, default=0.5, help='initial pendulum angle (rad)')
parser.add_argument('--theta_ref', type=float, default=np.pi, help='reference angle (rad), default upright (pi)')
args = parser.parse_args()

# initial states
theta = args.theta0  # initial angle (rad)
theta_dot = 0.0

# controller
ctrl = AdaptiveController(k_p=15.0, k_d=5.0, gamma=5.0)
ctrl.set_dt(dt)

# record history
theta_hist = np.zeros(steps)
theta_dot_hist = np.zeros(steps)
ut_hist = np.zeros(steps)
theta_hat_hist = np.zeros(steps)

for i in range(steps):
    # compute control torque
    ut = ctrl.control(theta, theta_dot, theta_ref=args.theta_ref)
    # pendulum dynamics: theta_ddot = -(g/l)*sin(theta) + ut/(m*l^2)
    theta_ddot = - (g / ctrl.l) * np.sin(theta) + ut / (ctrl.m * ctrl.l ** 2)

    # Euler integration
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    # record
    theta_hist[i] = theta
    theta_dot_hist[i] = theta_dot
    ut_hist[i] = ut
    theta_hat_hist[i] = ctrl.theta_hat

# plotting results
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(time, theta_hist)
plt.ylabel('Angle (rad)')
plt.title('Adaptive Control of Pendulum')

plt.subplot(3, 1, 2)
plt.plot(time, ut_hist)
plt.ylabel('Control torque (Nm)')

plt.subplot(3, 1, 3)
plt.plot(time, theta_hat_hist)
plt.ylabel('Parameter Estimate')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

# animation of pendulum
import matplotlib.animation as animation

fig2, ax2 = plt.subplots()
ax2.set_aspect('equal')
L = ctrl.l
ax2.set_xlim(-L, L)
ax2.set_ylim(-1.2*L, 1.2*L)
line, = ax2.plot([], [], 'o-', lw=2)

def init_anim():
    line.set_data([], [])
    return line,

def update_anim(frame):
    x = L * np.sin(theta_hist[frame])
    y = -L * np.cos(theta_hist[frame])
    line.set_data([0, x], [0, y])
    return line,

ani = animation.FuncAnimation(fig2, update_anim, frames=steps, init_func=init_anim, interval=dt*1000, blit=True)
plt.show()