# -*- coding: utf-8 -*-
"""
Lyapunov-Based Control for Nonholonomic Two-Wheeled Mobile Robot

This script implements a simulation environment for a two-wheeled mobile robot
controlled using a Lyapunov-based approach with a switching strategy to a PID
controller near the target. It includes classes for the robot (Plant),
the controllers (Controller, LyapunovController, ControllerPID), the simulation
loop (Simulation), and animation (Animator).

Default execution saves the animation as 'robot_simulation.gif'.
Requires libraries: numpy, matplotlib, tqdm, Pillow (for GIF saving).
For HTML5 video output (commented out), ffmpeg installation is required.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Arrow, FancyArrowPatch
from IPython.display import HTML, display # Keep for potential notebook use
from tqdm.notebook import tqdm # Use notebook version for better Colab/Jupyter display
import time
import math
import sys # To check if running in interactive environment

# --------------------------------------------------------------------------
# Mathematical Derivation (Lyapunov-Based Control - Polar Coordinates)
# --------------------------------------------------------------------------
# (Derivation comments remain the same as previous version)
# ... [omitted for brevity, see previous response if needed] ...
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Code Implementation
# --------------------------------------------------------------------------

# --- Helper Function ---
def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

# --- Plant ---
class Plant:
    """Abstract base class for a system (plant)."""
    def __init__(self):
        self.state = None
        self.dt = 0.01 # Default timestep

    def step(self, action):
        """
        Update state based on current state and action using Euler integration.
        Must be implemented by subclasses.

        Args:
            action: Control input.

        Returns:
            next_state: The updated state after applying the action.
        """
        raise NotImplementedError("Subclass must implement abstract method 'step'")

    def reset(self, initial_state=None):
        """
        Reset plant to a given initial state or a default state.

        Args:
            initial_state: The state to reset to (optional).

        Returns:
            The initial state after reset.
        """
        self.state = initial_state if initial_state is not None else np.zeros_like(self.state)
        return self.state

class TwoWheeledRobot(Plant):
    """
    Represents a two-wheeled differential drive mobile robot.

    State: [x, y, theta] (position and heading angle)
    Action: [v, omega] (linear and angular velocity)
    """
    def __init__(self, dt=0.01, wheel_radius=0.05, robot_width=0.3, v_max=1.0, omega_max=np.pi/2):
        """
        Initializes the robot.

        Args:
            dt (float): Simulation timestep in seconds.
            wheel_radius (float): Radius of the wheels in meters.
            robot_width (float): Distance between the two wheels in meters.
            v_max (float): Maximum linear velocity (m/s).
            omega_max (float): Maximum angular velocity (rad/s).
        """
        super().__init__()
        self.dt = dt
        self.r = wheel_radius
        self.L = robot_width / 2  # Half distance between wheels
        self.v_max = v_max
        self.omega_max = omega_max

        # Initial state [x, y, theta]
        self.state = np.zeros(3)

    def step(self, action):
        """
        Updates the robot's state using its kinematic model and Euler integration.

        Args:
            action (np.ndarray): Control input [v, omega].

        Returns:
            np.ndarray: The next state [x_next, y_next, theta_next].
        """
        if self.state is None:
            raise ValueError("Robot state is not initialized. Call reset() first.")
        if not isinstance(action, np.ndarray) or action.shape != (2,):
             raise ValueError(f"Action must be a numpy array of shape (2,). Got: {action}")

        x, y, theta = self.state

        # Extract and clip control inputs
        v = np.clip(action[0], -self.v_max, self.v_max)
        omega = np.clip(action[1], -self.omega_max, self.omega_max)

        # Kinematic equations (Euler integration)
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt

        # Normalize angle to [-pi, pi]
        theta_next = normalize_angle(theta_next)

        # Update state
        self.state = np.array([x_next, y_next, theta_next])

        return self.state

    def reset(self, initial_state=None):
        """
        Resets the robot to a specified initial state or to [0, 0, 0].

        Args:
            initial_state (np.ndarray, optional): The state [x, y, theta] to reset to. Defaults to None ([0,0,0]).

        Returns:
            np.ndarray: The state after reset.
        """
        if initial_state is None:
            self.state = np.zeros(3)
        else:
             if not isinstance(initial_state, np.ndarray) or initial_state.shape != (3,):
                 raise ValueError("initial_state must be a numpy array of shape (3,)")
             self.state = np.array(initial_state)
             self.state[2] = normalize_angle(self.state[2]) # Ensure initial angle is normalized
        return self.state

    def get_wheel_velocities(self, v, omega):
        """
        Calculates the required angular velocities for left and right wheels.

        Args:
            v (float): Linear velocity.
            omega (float): Angular velocity.

        Returns:
            tuple: (omega_left, omega_right) wheel angular velocities in rad/s.
        """
        if self.r <= 0:
             raise ValueError("Wheel radius must be positive.")
        omega_r = (v + self.L * omega) / self.r
        omega_l = (v - self.L * omega) / self.r
        return omega_l, omega_r

# --- Controllers ---
class Controller:
    """Abstract base class for controllers."""
    def __init__(self):
        pass

    def get_action(self, state, target):
        """
        Computes the control action. Must be implemented by subclasses.

        Args:
            state (np.ndarray): Current state of the system.
            target (np.ndarray): Target state or setpoint.

        Returns:
            np.ndarray: The calculated control action.
        """
        raise NotImplementedError("Subclass must implement abstract method 'get_action'")

    def reset(self):
        """Resets any internal state of the controller (e.g., integral terms)."""
        pass # Default implementation does nothing


class ControllerPID(Controller):
    """
    A basic PID controller adapted for the mobile robot.
    It uses PID terms on the forward error (in robot frame) for 'v'
    and PID terms on the heading error for 'omega'.
    Note: This is a simplified PID structure for this context.
    """
    def __init__(self, k_p_v=1.0, k_i_v=0.0, k_d_v=0.0,
                 k_p_w=1.0, k_i_w=0.0, k_d_w=0.0, dt=0.01):
        """
        Initializes the PID controller gains.

        Args:
            k_p_v, k_i_v, k_d_v (float): PID gains for linear velocity 'v'.
            k_p_w, k_i_w, k_d_w (float): PID gains for angular velocity 'omega'.
            dt (float): Timestep, used for integral and derivative calculations.
        """
        super().__init__()
        self.kp_v, self.ki_v, self.kd_v = k_p_v, k_i_v, k_d_v
        self.kp_w, self.ki_w, self.kd_w = k_p_w, k_i_w, k_d_w
        self.dt = dt
        self.integral_v = 0.0
        self.integral_w = 0.0
        self.prev_error_v = 0.0
        self.prev_error_w = 0.0

    def reset(self):
        """Resets the integral terms and previous errors."""
        self.integral_v = 0.0
        self.integral_w = 0.0
        self.prev_error_v = 0.0
        self.prev_error_w = 0.0

    def get_action(self, state, target):
        """
        Computes the PID control action [v, omega].

        Args:
            state (np.ndarray): Current robot state [x, y, theta].
            target (np.ndarray): Target robot state [x_d, y_d, theta_d].

        Returns:
            np.ndarray: Control action [v, omega].
        """
        x, y, theta = state
        x_d, y_d, theta_d = target

        # --- Position Error (for v) ---
        error_x = x_d - x
        error_y = y_d - y
        # Project position error onto robot's forward direction
        error_v = error_x * np.cos(theta) + error_y * np.sin(theta) # Distance error along robot heading

        # --- Orientation Error (for omega) ---
        # Error 1: Angle to target point
        angle_to_target = np.arctan2(error_y, error_x)
        error_w1 = normalize_angle(angle_to_target - theta)
        # Error 2: Final orientation error
        error_w2 = normalize_angle(theta_d - theta)

        # Use angle to target primarily, blend in final orientation error near target
        rho = np.sqrt(error_x**2 + error_y**2)
        # Simple heuristic: prioritize pointing to target, then fix final angle
        # A more sophisticated approach might blend these errors based on distance.
        # Here, let's focus PID on the final orientation error `error_w2`
        # for fine-tuning near the goal, as the Lyapunov part handles pointing.
        error_w = error_w2 # Focus PID on final theta error

        # --- PID Calculation for v ---
        self.integral_v += error_v * self.dt
        derivative_v = (error_v - self.prev_error_v) / self.dt
        v = self.kp_v * error_v + self.ki_v * self.integral_v + self.kd_v * derivative_v
        self.prev_error_v = error_v

        # --- PID Calculation for omega ---
        self.integral_w += error_w * self.dt
        derivative_w = (error_w - self.prev_error_w) / self.dt
        omega = self.kp_w * error_w + self.ki_w * self.integral_w + self.kd_w * derivative_w
        self.prev_error_w = error_w

        return np.array([v, omega])


class LyapunovController(Controller):
    """
    Lyapunov-based controller for the two-wheeled robot using polar coordinates
    with a switching strategy to a PID controller near the target.
    """
    def __init__(self, k_rho=1.0, k_alpha=3.0, k_beta=0.8, switch_distance=0.1, pid_gains=None, dt=0.01):
        """
        Initializes the Lyapunov controller.

        Args:
            k_rho (float): Gain for linear velocity control (proportional to distance).
            k_alpha (float): Gain for steering towards the target direction.
            k_beta (float): Gain for correcting the final orientation.
            switch_distance (float): Distance (rho) below which to switch to PID.
            pid_gains (dict, optional): Gains for the PID controller used near target.
                                        Defaults to basic gains if None.
                                        Example: {'k_p_v': 0.8, 'k_i_v': 0.1, ...}
            dt (float): Timestep, passed to the internal PID controller.
        """
        super().__init__()
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        self.switch_distance = switch_distance

        # Default PID gains if none provided
        if pid_gains is None:
            pid_gains = {'k_p_v': 0.8, 'k_i_v': 0.01, 'k_d_v': 0.05,
                         'k_p_w': 1.5, 'k_i_w': 0.05, 'k_d_w': 0.1, 'dt': dt}
        else:
             pid_gains['dt'] = dt # Ensure dt is passed

        self.pid_controller = ControllerPID(**pid_gains)
        self._use_pid = False # Internal flag to track which controller is active

    def reset(self):
        """Resets the internal PID controller."""
        self.pid_controller.reset()
        self._use_pid = False

    def get_action(self, state, target):
        """
        Computes the control action [v, omega] using Lyapunov or PID.

        Args:
            state (np.ndarray): Current robot state [x, y, theta].
            target (np.ndarray): Target robot state [x_d, y_d, theta_d].

        Returns:
            np.ndarray: Control action [v, omega].
        """
        x, y, theta = state
        x_d, y_d, theta_d = target

        # Calculate error components
        error_x = x_d - x
        error_y = y_d - y
        rho = np.sqrt(error_x**2 + error_y**2)

        # --- Switching Logic ---
        if rho < self.switch_distance:
            self._use_pid = True
            # Ensure PID is reset only once when switching
            if not hasattr(self, '_pid_just_switched') or not self._pid_just_switched:
                    self.pid_controller.reset()
                    self._pid_just_switched = True
            return self.pid_controller.get_action(state, target)
        else:
            self._use_pid = False
            self._pid_just_switched = False # Reset switch flag

        # --- Lyapunov Control Law (Polar Coordinates) ---
        # alpha: angle error (robot heading vs target direction)
        angle_to_target = np.arctan2(error_y, error_x)
        alpha = normalize_angle(angle_to_target - theta)

        # beta: orientation error (target orientation vs target direction)
        # Using the form beta = target_angle - current_angle - alpha
        # Note: The sign of k_beta might need adjustment based on this definition and desired behavior.
        # If beta = theta_d - angle_to_target, the gain might behave differently.
        beta = normalize_angle(theta_d - theta - alpha)

        # Control laws (derived from stability analysis)
        v = self.k_rho * rho * np.cos(alpha)

        # Combine steering towards target (alpha) and final orientation correction (beta)
        omega = self.k_alpha * alpha + self.k_beta * beta

        return np.array([v, omega])

# --- Simulation ---
class Simulation:
    """Handles the simulation loop and data storage."""
    def __init__(self, plant: Plant, controller: Controller, total_time=10.0):
        """
        Initializes the simulation environment.

        Args:
            plant (Plant): The system to be controlled.
            controller (Controller): The controller providing actions.
            total_time (float): Total simulation time in seconds.
        """
        if not isinstance(plant, Plant):
             raise TypeError("plant must be an instance of Plant or its subclass.")
        if not isinstance(controller, Controller):
             raise TypeError("controller must be an instance of Controller or its subclass.")

        self.plant = plant
        self.controller = controller
        self.total_time = total_time
        self.dt = plant.dt
        if self.dt <= 0:
            raise ValueError("Plant dt must be positive.")
        self.num_steps = int(total_time / self.dt)

        # Data storage initialized in run()
        self.states = None
        self.actions = None
        self.times = None
        self.controller_type = None # To track Lyapunov vs PID usage

    def run(self, initial_state, target_state):
        """
        Runs the simulation from an initial state to a target state.

        Args:
            initial_state (np.ndarray): Initial state of the plant [x, y, theta].
            target_state (np.ndarray): Target state [x_d, y_d, theta_d].
        """
        print(f"Starting simulation: {self.total_time}s, dt={self.dt}s, {self.num_steps} steps")
        # Reset plant and controller
        current_state = self.plant.reset(initial_state)
        self.controller.reset()

        # Initialize storage arrays (+1 for initial state)
        num_states = len(current_state)
        self.states = np.zeros((self.num_steps + 1, num_states))
        self.actions = np.zeros((self.num_steps, 2))  # Assuming action is [v, omega]
        self.times = np.linspace(0, self.total_time, self.num_steps + 1)
        self.controller_type = np.zeros(self.num_steps) # 0: Lyapunov, 1: PID

        # Store initial state
        self.states[0] = current_state

        # Simulation loop with progress bar
        # Use plain tqdm if not in notebook/interactive environment
        looper = tqdm(range(self.num_steps), desc="Simulating") if hasattr(sys, 'ps1') else range(self.num_steps)

        for i in looper:
            # 1. Get control action
            action = self.controller.get_action(current_state, target_state)

            # Store controller type if available
            if isinstance(self.controller, LyapunovController):
                self.controller_type[i] = 1 if self.controller._use_pid else 0

            # 2. Apply action to plant
            next_state = self.plant.step(action)

            # 3. Store data
            self.actions[i] = action
            self.states[i+1] = next_state

            # 4. Update current state
            current_state = next_state

            # Optional: Termination condition
            # ... (termination logic can be added here) ...

        print("Simulation finished.")


    def plot_results(self, target_state=None):
        """Plots the simulation results (trajectory, states, actions)."""
        if self.states is None or self.actions is None or self.times is None:
            print("Error: Simulation data not available. Run simulation first.")
            return

        # Check if running in non-interactive mode, switch backend if necessary
        # Needs testing, might not always work or be necessary
        # try:
        #     plt.figure()
        # except Exception: # A bit broad, but catches environments without GUI
        #     print("Non-interactive environment detected, switching Matplotlib backend to 'Agg'.")
        #     import matplotlib
        #     matplotlib.use('Agg')
        #     import matplotlib.pyplot as plt

        plt.style.use('seaborn-v0_8-whitegrid') # Nicer style
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Simulation Results', fontsize=16)

        # Plot Trajectory (axs[0, 0])
        ax = axs[0, 0]
        ax.plot(self.states[:, 0], self.states[:, 1], 'b-', lw=2, label='Robot Trajectory')
        ax.plot(self.states[0, 0], self.states[0, 1], 'go', markersize=10, label='Start')
        if target_state is not None:
            ax.plot(target_state[0], target_state[1], 'rx', markersize=12, mew=2, label='Target Position')
            # Plot target orientation
            dx = 0.5 * np.cos(target_state[2])
            dy = 0.5 * np.sin(target_state[2])
            ax.arrow(target_state[0], target_state[1], dx, dy, head_width=0.15, head_length=0.2, fc='r', ec='r', label='Target Orientation')
        ax.plot(self.states[-1, 0], self.states[-1, 1], 'mo', markersize=8, alpha=0.8, label='Final Position')
        ax.set_xlabel('X Position [m]')
        ax.set_ylabel('Y Position [m]')
        ax.set_title('Robot Trajectory')
        ax.axis('equal')
        ax.legend(fontsize=9)
        ax.grid(True)

        # Plot Position States vs Time (axs[0, 1])
        ax = axs[0, 1]
        ax.plot(self.times, self.states[:, 0], 'r-', label='x(t)')
        ax.plot(self.times, self.states[:, 1], 'g-', label='y(t)')
        if target_state is not None:
            ax.axhline(target_state[0], color='r', linestyle='--', alpha=0.5, label='x_target')
            ax.axhline(target_state[1], color='g', linestyle='--', alpha=0.5, label='y_target')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')
        ax.set_title('Position vs Time')
        ax.legend(fontsize=9)
        ax.grid(True)

        # Plot Orientation State vs Time (axs[1, 0])
        ax = axs[1, 0]
        ax.plot(self.times, np.degrees(self.states[:, 2]), 'b-', label='θ(t)') # Plot in degrees
        if target_state is not None:
            ax.axhline(np.degrees(target_state[2]), color='b', linestyle='--', alpha=0.5, label='θ_target')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Orientation [deg]')
        ax.set_title('Orientation vs Time')
        ax.legend(fontsize=9)
        ax.grid(True)

        # Plot Control Actions vs Time (axs[1, 1])
        ax = axs[1, 1]
        action_times = self.times[:-1] # Actions applied between states
        ax.plot(action_times, self.actions[:, 0], 'm-', label='v(t)')
        ax_twin = ax.twinx() # Use twin axes for omega if scales differ
        ax_twin.plot(action_times, self.actions[:, 1], 'c-', label='ω(t)')

        # Indicate controller switching if data is available
        if hasattr(self, 'controller_type') and self.controller_type is not None:
            switches = np.where(np.diff(self.controller_type))[0]
            for switch_idx in switches:
                switch_time = action_times[switch_idx]
                ax.axvline(switch_time, color='k', linestyle=':', lw=1, alpha=0.7, label='_nolegend_')
            pid_regions = np.where(self.controller_type == 1)[0]
            if len(pid_regions)>0:
                ax.text(0.95, 0.05, 'PID active near end', transform=ax.transAxes,
                        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))


        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Linear Velocity (v) [m/s]', color='m')
        ax_twin.set_ylabel('Angular Velocity (ω) [rad/s]', color='c')
        ax.set_title('Control Inputs vs Time')
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=9)
        ax.grid(True)


        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
        plt.show() # Display the plot


# --- Animation ---
class Animator:
    """Creates an HTML5 animation or saves to file."""
    def __init__(self, simulation: Simulation, target_state=None, robot_radius=0.15):
        """
        Initializes the Animator.

        Args:
            simulation (Simulation): The Simulation object containing trajectory data.
            target_state (np.ndarray, optional): Target state [x_d, y_d, theta_d] for visualization.
            robot_radius (float): Radius of the circle representing the robot in the animation.
        """
        if not isinstance(simulation, Simulation):
             raise TypeError("simulation must be an instance of Simulation.")
        if simulation.states is None:
             raise ValueError("Simulation data is not available. Run simulation first.")

        self.simulation = simulation
        self.target_state = target_state
        self.robot_radius = robot_radius
        self.dt = simulation.dt

        # Animation objects (will be updated)
        self.fig, self.ax = None, None
        self.robot_body = None
        self.robot_direction = None
        self.target_marker = None
        self.target_arrow = None
        self.path_line = None
        self.v_arrow = None
        self.w_arrow_patch = None # Use Arc for omega visualization
        self.time_text = None

    def _init_animation(self):
        """Initializes the static elements of the animation plot."""
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Set plot limits with margin
        all_x = self.simulation.states[:, 0]
        all_y = self.simulation.states[:, 1]
        margin = max(1.0, self.robot_radius * 5) # Ensure margin is reasonable
        x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
        y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin
        # Include target in limits if provided
        if self.target_state is not None:
             x_min = min(x_min, self.target_state[0] - margin)
             x_max = max(x_max, self.target_state[0] + margin)
             y_min = min(y_min, self.target_state[1] - margin)
             y_max = max(y_max, self.target_state[1] + margin)

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('X Position [m]')
        self.ax.set_ylabel('Y Position [m]')
        self.ax.set_title('Robot Simulation Animation')
        self.ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
        self.ax.grid(True)

        # Plot full trajectory faintly
        self.ax.plot(all_x, all_y, 'b--', lw=1, alpha=0.4, label='Full Path')

        # Plot start and target markers
        self.ax.plot(all_x[0], all_y[0], 'go', markersize=8, label='Start')
        if self.target_state is not None:
            self.target_marker = self.ax.plot(self.target_state[0], self.target_state[1],
                                             'rx', markersize=10, mew=2, label='Target Pos')[0]
            dx = 0.5 * np.cos(self.target_state[2])
            dy = 0.5 * np.sin(self.target_state[2])
            self.target_arrow = self.ax.arrow(self.target_state[0], self.target_state[1], dx, dy,
                                                 head_width=0.15, head_length=0.2, fc='r', ec='r', alpha=0.6, label='Target Ori')

        # Initialize dynamic elements (robot, path trace, arrows)
        self.robot_body = Circle((all_x[0], all_y[0]), self.robot_radius,
                                     edgecolor='black', facecolor='lightblue', zorder=10)
        self.ax.add_patch(self.robot_body)

        dir_len = self.robot_radius * 1.1
        theta0 = self.simulation.states[0, 2]
        self.robot_direction = self.ax.arrow(all_x[0], all_y[0],
                                                 dir_len * np.cos(theta0), dir_len * np.sin(theta0),
                                                 head_width=0.1, head_length=0.15, fc='red', ec='red', zorder=11, label='Heading')

        # Line object for tracing the path dynamically
        self.path_line, = self.ax.plot([], [], 'b-', lw=2, label='Current Path', zorder=5)

        # Velocity arrow
        self.v_arrow = FancyArrowPatch((0,0), (0,0), color='green', arrowstyle='-|>', mutation_scale=15, lw=1.5, label='Velocity (v)', zorder=12)
        self.ax.add_patch(self.v_arrow)

        # Angular velocity arc (using Arc patch)
        self.w_arrow_patch = FancyArrowPatch((0, 0), (0, 0), connectionstyle="arc3,rad=0",
                                             color="orange", arrowstyle="-|>", mutation_scale=15, lw=1.5, label='Ang. Vel. (ω)', zorder=12)

        self.ax.add_patch(self.w_arrow_patch)


        # Time display
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10,
                                            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

        self.ax.legend(fontsize=9, loc='upper right')

        # Return list of artists that need updating in animate function
        return (self.robot_body, self.robot_direction, self.path_line, self.time_text, self.v_arrow, self.w_arrow_patch)

    def _animate(self, i):
        """Updates the animation elements for frame i."""
        # Get state and action for frame i
        x, y, theta = self.simulation.states[i]
        if i < len(self.simulation.actions):
            v, omega = self.simulation.actions[i]
        else: # For the very last frame, reuse last action
            v, omega = self.simulation.actions[-1]

        # Update robot position
        self.robot_body.center = (x, y)

        # Update robot direction arrow
        self.robot_direction.remove() # Remove old arrow
        dir_len = self.robot_radius * 1.1
        self.robot_direction = self.ax.arrow(x, y, dir_len * np.cos(theta), dir_len * np.sin(theta),
                                                 head_width=0.1, head_length=0.15, fc='red', ec='red', zorder=11)

        # Update traced path
        self.path_line.set_data(self.simulation.states[:i+1, 0], self.simulation.states[:i+1, 1])

        # Update velocity arrow
        v_scale = 0.5 # Scale factor for visualization
        v_dx = v_scale * v * np.cos(theta)
        v_dy = v_scale * v * np.sin(theta)
        # Use set_positions for FancyArrowPatch
        self.v_arrow.set_positions((x, y), (x + v_dx, y + v_dy))
        self.v_arrow.set_visible(abs(v) > 0.01) # Hide if velocity is near zero

        # Update angular velocity arc
        if abs(omega) > 0.05: # Only show if significant
            arc_radius = self.robot_radius * 1.3
            start_angle_deg = np.degrees(theta)
            # Define arc extent based on omega sign and magnitude (visual scaling)
            arc_angle_deg = np.clip(omega * 40, -80, 80) # Scaled angle extent
            # Position the arc around the robot body
            # Calculate start and end points for the arc's FancyArrowPatch
            start_pt_arc = (x + arc_radius * np.cos(theta), y + arc_radius * np.sin(theta))
            end_angle_rad = theta + np.radians(arc_angle_deg)
            end_pt_arc = (x + arc_radius * np.cos(end_angle_rad), y + arc_radius * np.sin(end_angle_rad))

            # Set connection style dynamically based on omega sign
            rad_sign = 0.4 * np.sign(omega)
            self.w_arrow_patch.set_connectionstyle(f"arc3,rad={rad_sign:.2f}")
            self.w_arrow_patch.set_positions(start_pt_arc, end_pt_arc)
            self.w_arrow_patch.set_visible(True)
        else:
            self.w_arrow_patch.set_visible(False)


        # Update time text
        current_time = self.simulation.times[i]
        self.time_text.set_text(f'Time: {current_time:.2f} s')

        # Return list of updated artists
        return (self.robot_body, self.robot_direction, self.path_line, self.time_text, self.v_arrow, self.w_arrow_patch)

    def create_animation(self, interval=50, filename=None):
        """
        Creates the animation, saving to file or returning HTML object.

        Args:
            interval (int): Delay between frames in milliseconds.
            filename (str, optional): If provided, saves the animation to this file (e.g., 'robot_sim.gif' or 'robot_sim.mp4').
                                        Requires Pillow (for gif) or ffmpeg (for mp4) installed.
                                        If None, attempts to return HTML5 video (requires ffmpeg).

        Returns:
            IPython.display.HTML or None: HTML5 video object for display in notebooks if filename is None and ffmpeg is available.
                                         Returns None if filename is provided or if HTML5 generation fails.
        """
        if self.fig is None:
            self._init_animation() # Initialize plot if not already done

        num_frames = len(self.simulation.states)

        # Create animation
        anim = FuncAnimation(self.fig, self._animate, init_func=self._init_animation,
                                     frames=num_frames, interval=interval, blit=False)

        output = None
        try:
            if filename:
                print(f"Saving animation to {filename}...")
                start_save_time = time.time()
                if filename.lower().endswith('.gif'):
                    # Requires Pillow
                    writer = PillowWriter(fps=int(1000 / interval))
                    anim.save(filename, writer=writer)
                elif filename.lower().endswith('.mp4'):
                    # Requires ffmpeg
                    try:
                        plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg' # Use ffmpeg from PATH
                        writer = 'ffmpeg'
                        anim.save(filename, writer=writer, fps=int(1000/interval), dpi=150)
                    except FileNotFoundError:
                        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH to save as MP4.")
                else:
                    print(f"Warning: Unknown file extension for saving animation: {filename}. Supported extensions are .gif and .mp4.")
                end_save_time = time.time()
                print(f"Animation saved in {end_save_time - start_save_time:.2f} seconds.")
            else:
                # Try to display as HTML5 video (requires ffmpeg)
                try:
                    output = anim.to_html5_video()
                except Exception as e:
                    print(f"Error creating HTML5 video (requires ffmpeg): {e}")
                    print("Consider saving to a file (e.g., .gif or .mp4).")
        except Exception as e:
            print(f"An error occurred during animation creation/saving: {e}")

        return output

# --- Main Execution ---
if __name__ == '__main__':
    # Simulation parameters
    total_time = 20.0 # Increased simulation time
    dt = 0.01

    # Robot parameters
    robot_radius = 0.15
    robot_width = 0.3
    v_max = 1.5 # Increased max velocity
    omega_max = np.pi # Increased max angular velocity

    # Initial and target states
    initial_state = np.array([0.0, 0.0, 0.0])
    target_state = np.array([5.0, 5.0, np.pi/2]) # Target with orientation

    # Lyapunov controller parameters
    lyapunov_gains = {'k_rho': 1.2, 'k_alpha': 3.5, 'k_beta': -0.6, 'switch_distance': 0.2} # Adjusted gains
    pid_gains = {'k_p_v': 1.0, 'k_i_v': 0.02, 'k_d_v': 0.1,
                 'k_p_w': 2.0, 'k_i_w': 0.1, 'k_d_w': 0.2, 'dt': dt} # Adjusted PID gains
    lyapunov_controller = LyapunovController(dt=dt, pid_gains=pid_gains, **lyapunov_gains)

    # Initialize plant and simulation
    robot = TwoWheeledRobot(dt=dt, wheel_radius=robot_radius / 3, robot_width=robot_width, v_max=v_max, omega_max=omega_max)
    simulation = Simulation(plant=robot, controller=lyapunov_controller, total_time=total_time)

    # Run the simulation
    simulation.run(initial_state, target_state)

    # Plot the results
    simulation.plot_results(target_state=target_state)

    # Create and save the animation
    animator = Animator(simulation, target_state=target_state, robot_radius=robot_radius)
    animation_filename = 'robot_simulation.mp4' # Changed to .mp4
    html_output = animator.create_animation(interval=50, filename=animation_filename)

    # Optionally display the animation in a notebook environment
    if html_output:
        display(HTML(html_output))