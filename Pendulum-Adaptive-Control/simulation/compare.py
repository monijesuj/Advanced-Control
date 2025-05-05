import os, sys
# add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import argparse
from controllers.adaptive_controller import AdaptiveController
from controllers.lyapunov_controller import LyapunovController

# parse CLI args
parser = argparse.ArgumentParser(description='Compare PD (Lyapunov) vs MRAC adaptive control under friction')
parser.add_argument('--theta0', type=float, default=0.5, help='initial angle (rad)')
parser.add_argument('--theta_ref', type=float, default=np.pi, help='target angle (rad)')
parser.add_argument('--b', type=float, default=1.0, help='friction coefficient (Nms)')
parser.add_argument('--dt', type=float, default=0.001, help='time step')
parser.add_argument('--T', type=float, default=10.0, help='total simulation time')
args = parser.parse_args()

# parameters
g = 9.81
m, l = 1.0, 1.0
b = args.b

dt, T = args.dt, args.T
steps = int(T/dt)
time = np.linspace(0, T, steps)

# initialize controllers
pd_ctrl = LyapunovController(m=m, l=l, k_p=15, k_d=5)
ad_ctrl = AdaptiveController(m=m, l=l, k_p=15, k_d=5, gamma=5)
ad_ctrl.set_dt(dt)

# histories
theta_pd = np.zeros(steps)
theta_ad = np.zeros(steps)
theta_pd_dot = np.zeros(steps)
theta_ad_dot = np.zeros(steps)

# initial states
th_pd = th_ad = args.theta0
th_pd_dot = th_ad_dot = 0.0

for i in range(steps):
    # PD control
    u_pd = pd_ctrl.control(th_pd, th_pd_dot, args.theta_ref)
    tau_fric_pd = -b * th_pd_dot
    theta_ddot_pd = -(g/l)*np.sin(th_pd) + (u_pd + tau_fric_pd)/(m*l**2)
    th_pd_dot += theta_ddot_pd*dt
    th_pd += th_pd_dot*dt
    theta_pd[i], theta_pd_dot[i] = th_pd, th_pd_dot
    # adaptive control
    u_ad = ad_ctrl.control(th_ad, th_ad_dot, args.theta_ref)
    tau_fric_ad = -b * th_ad_dot
    theta_ddot_ad = -(g/l)*np.sin(th_ad) + (u_ad + tau_fric_ad)/(m*l**2)
    th_ad_dot += theta_ddot_ad*dt
    th_ad += th_ad_dot*dt
    theta_ad[i], theta_ad_dot[i] = th_ad, th_ad_dot

# plot
plt.figure(figsize=(8,6))
plt.plot(time, theta_pd, label='Lyapunov (PD)')
plt.plot(time, theta_ad, label='Adaptive (MRAC)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title(f'Comparison with friction b={b}')
plt.legend()
plt.grid(True)
plt.show()

# animation comparison of both controllers
import matplotlib.animation as animation

fig_anim, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
axes[0].set_title('Lyapunov PD')
axes[1].set_title('Adaptive MRAC')
for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlim(-l, l)
    ax.set_ylim(-1.2*l, 1.2*l)

line_pd, = axes[0].plot([], [], 'o-', lw=2)
line_ad, = axes[1].plot([], [], 'o-', lw=2)

# initialization function
def init_anim():
    line_pd.set_data([], [])
    line_ad.set_data([], [])
    return line_pd, line_ad

# update function for animation
def update_anim(i):
    # PD pendulum
    x_pd = l * np.sin(theta_pd[i])
    y_pd = -l * np.cos(theta_pd[i])
    line_pd.set_data([0, x_pd], [0, y_pd])
    # adaptive pendulum
    x_ad = l * np.sin(theta_ad[i])
    y_ad = -l * np.cos(theta_ad[i])
    line_ad.set_data([0, x_ad], [0, y_ad])
    return line_pd, line_ad

ani = animation.FuncAnimation(
    fig_anim, update_anim, frames=steps,
    init_func=init_anim, interval=dt*1000, blit=True)
plt.show()