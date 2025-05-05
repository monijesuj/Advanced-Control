import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np

class Visualizer:
    def __init__(self, desired_path):
        """ Initialize the visualizer. Sets up the plot structure. """
        self.desired_path = desired_path
        self.fig = plt.figure(figsize=(20, 12)) # Larger figure

        # Create a grid for subplots
        grid = plt.GridSpec(6, 2, height_ratios=[3, 1, 1, 1, 1, 1], hspace=0.8, wspace=0.3) # Add wspace

        # --- Column 1: Path Plot ---
        self.path_ax = self.fig.add_subplot(grid[:, 0])
        self.path_ax.set_aspect('equal', adjustable='box')
        self.path_ax.set_title("Robot Path Tracking (Adaptive Dynamic Control)")
        self.path_ax.set_xlabel("X-axis (m)")
        self.path_ax.set_ylabel("Y-axis (m)")
        self.path_ax.plot(desired_path[:, 0], desired_path[:, 1], 'r--', label='Desired Path', linewidth=1.5)
        self.actual_path_line, = self.path_ax.plot([], [], 'g-', label='Actual Path', linewidth=1.5)
        self.robot_marker, = self.path_ax.plot([], [], 'bo', markersize=8, label="Robot Position")
        self.robot_arrow = patches.Arrow(0, 0, 0, 0, width=0.2, color='blue', zorder=5) # Ensure robot arrow on top
        self.path_ax.add_patch(self.robot_arrow)
        self.velocity_arrow = patches.Arrow(0, 0, 0, 0, width=0.1, color='lime', alpha=0.7, zorder=4) # Velocity arrow below
        self.path_ax.add_patch(self.velocity_arrow)
        self.path_ax.legend(loc='best') # Changed legend location
        self.path_ax.grid(True)

        # --- Column 2: Time Plots ---
        # Robot Velocities (Actual vs Desired)
        self.velocity_ax = self.fig.add_subplot(grid[0, 1])
        self.velocity_ax.set_title("Robot Velocities")
        self.velocity_ax.set_ylabel("Velocity")
        self.v1_actual_line, = self.velocity_ax.plot([], [], 'r-', label="$v_1$ Actual (m/s)")
        self.omega_actual_line, = self.velocity_ax.plot([], [], 'm-', label="$\\omega$ Actual (rad/s)")
        self.v1_desired_line, = self.velocity_ax.plot([], [], 'r:', label="$v_{1d}$ Desired")
        self.omega_desired_line, = self.velocity_ax.plot([], [], 'm:', label="$\\omega_d$ Desired")
        self.velocity_ax.legend(fontsize='small') # Smaller font
        self.velocity_ax.grid(True)
        self.velocity_ax.tick_params(axis='x', labelbottom=False)

        # Control Torques
        self.torque_ax = self.fig.add_subplot(grid[1, 1], sharex=self.velocity_ax)
        self.torque_ax.set_title("Control Torques (Cmd)")
        self.torque_ax.set_ylabel("Torque (Nm)")
        self.tau_left_line, = self.torque_ax.plot([], [], 'b-', label="$\\tau_L$ Left")
        self.tau_right_line, = self.torque_ax.plot([], [], 'g-', label="$\\tau_R$ Right")
        self.torque_ax.legend(fontsize='small')
        self.torque_ax.grid(True)
        self.torque_ax.tick_params(axis='x', labelbottom=False)

        # Kinematic Errors (from Kinematic Controller)
        self.kin_error_ax = self.fig.add_subplot(grid[2, 1], sharex=self.velocity_ax)
        self.kin_error_ax.set_title("Kinematic Errors (Path Following)")
        self.kin_error_ax.set_ylabel("Error")
        self.error_fwd_line, = self.kin_error_ax.plot([], [], 'c-', label="Fwd Err (m)")
        self.error_lat_line, = self.kin_error_ax.plot([], [], 'y-', label="Lat Err (m)")
        self.error_theta_line, = self.kin_error_ax.plot([], [], 'k-', label="$\\theta$ Err (rad)")
        self.kin_error_ax.legend(fontsize='small')
        self.kin_error_ax.grid(True)
        self.kin_error_ax.tick_params(axis='x', labelbottom=False)

        # Velocity Errors (eta)
        self.vel_error_ax = self.fig.add_subplot(grid[3, 1], sharex=self.velocity_ax)
        self.vel_error_ax.set_title("Velocity Errors ($\\eta = v - v_d$)")
        self.vel_error_ax.set_ylabel("Vel Error")
        self.eta1_line, = self.vel_error_ax.plot([], [], 'r-', label="$\\eta_1 = v_1 - v_{1d}$")
        self.eta2_line, = self.vel_error_ax.plot([], [], 'm-', label="$\\eta_2 = \\omega - \\omega_d$")
        self.vel_error_ax.legend(fontsize='small')
        self.vel_error_ax.grid(True)
        self.vel_error_ax.tick_params(axis='x', labelbottom=False)

        # Parameter Estimates (p_hat)
        self.param_ax = self.fig.add_subplot(grid[4, 1], sharex=self.velocity_ax)
        self.param_ax.set_title("Parameter Estimates ($\\hat{p}$)")
        self.param_ax.set_ylabel("Value")
        self.m_hat_line, = self.param_ax.plot([], [], 'b-', label="$\\hat{m}$ (mass)")
        self.I_hat_line, = self.param_ax.plot([], [], 'g-', label="$\\hat{I}$ (inertia)")
        self.m_true_line, = self.param_ax.plot([], [], 'b--', label="$m$ True", alpha=0.5)
        self.I_true_line, = self.param_ax.plot([], [], 'g--', label="$I$ True", alpha=0.5)
        self.param_ax.legend(fontsize='small')
        self.param_ax.grid(True)
        self.param_ax.tick_params(axis='x', labelbottom=False) # Hide for last plot too? No, keep it for last.

        # Disturbance plot
        self.dist_ax = self.fig.add_subplot(grid[5, 1], sharex=self.velocity_ax)
        self.dist_ax.set_title("Disturbances (Effective Torque)")
        self.dist_ax.set_xlabel("Time (s)")
        self.dist_ax.set_ylabel("Disturbance")
        self.dist1_line, = self.dist_ax.plot([], [], 'r-', label="$\\tau_{d1}$ ($v_1$ eq)")
        self.dist2_line, = self.dist_ax.plot([], [], 'm-', label="$\\tau_{d2}$ ($\\omega$ eq)")
        self.dist_ax.legend(fontsize='small')
        self.dist_ax.grid(True)


        # self.fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # tight_layout often fails with GridSpec
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.9, wspace=0.3) # Manual adjustment

        # Animation related attributes
        self.sim_data = None
        self.controller_status_list = None
        self.anim_step = 1
        self.true_m = None
        self.true_I = None
        self.artists = [] # Store artists for blitting


    def _init_animation(self):
        """ Initialize elements for animation drawing """
        # Reset lines
        self.artists = [self.actual_path_line, self.robot_marker,
                 self.v1_actual_line, self.omega_actual_line, self.v1_desired_line, self.omega_desired_line,
                 self.tau_left_line, self.tau_right_line,
                 self.error_fwd_line, self.error_lat_line, self.error_theta_line,
                 self.eta1_line, self.eta2_line,
                 self.m_hat_line, self.I_hat_line,
                 self.m_true_line, self.I_true_line,
                 self.dist1_line, self.dist2_line]
        for line in self.artists:
            if isinstance(line, plt.Line2D): # Ensure it's a line object
                 line.set_data([], [])
            elif isinstance(line, patches.Patch): # Handle patches like markers if needed
                 pass # Marker position set later

        # Reset arrows
        if hasattr(self.robot_arrow, 'axes') and self.robot_arrow.axes: self.robot_arrow.remove()
        if hasattr(self.velocity_arrow, 'axes') and self.velocity_arrow.axes: self.velocity_arrow.remove()

        # Recreate arrows for init state (position 0,0, angle 0)
        self.robot_arrow = patches.Arrow(0, 0, 0.01, 0, width=0.2, color='blue', zorder=5) # Small initial arrow
        self.velocity_arrow = patches.Arrow(0, 0, 0, 0, width=0.1, color='lime', alpha=0.7, zorder=4)
        self.path_ax.add_patch(self.robot_arrow)
        self.path_ax.add_patch(self.velocity_arrow)
        self.artists.extend([self.robot_arrow, self.velocity_arrow]) # Add arrows to artist list

        # Set dynamic plot limits based on full data
        max_time = self.sim_data["time"][-1] if len(self.sim_data["time"]) > 0 else 1.0

        # Extract data for limits (handle potential length mismatches if sim ended early)
        num_steps_completed = len(self.controller_status_list)
        num_time_points = len(self.sim_data['time'])

        vels_actual = self.sim_data["robot_vels"] # Length N+1
        vels_desired = np.array([s['v_d'] for s in self.controller_status_list]) if self.controller_status_list else np.zeros((0,2)) # Length N
        all_vels_range = vels_actual

        torques = self.sim_data["torques_cmd"][1:num_steps_completed+1] # Length N
        kin_errors = np.array([s['kin_errors'] for s in self.controller_status_list]) if self.controller_status_list else np.zeros((0,3)) # N
        vel_errors = np.array([s['eta'] for s in self.controller_status_list]) if self.controller_status_list else np.zeros((0,2)) # N
        params_hat = np.array([s['p_hat'] for s in self.controller_status_list]) if self.controller_status_list else np.zeros((0,2)) # N
        disturbances = self.sim_data["disturbances"][1:num_steps_completed+1] # Length N

        # --- Calculate limits function (safe) ---
        def get_lims(data, pad_factor=0.1, zero_center=False):
            if not isinstance(data, np.ndarray): data = np.array(data)
            if data.size == 0: return (-0.1, 0.1)
            finite_data = data[np.isfinite(data)]
            if finite_data.size == 0: return (-0.1, 0.1)
            min_val, max_val = np.min(finite_data), np.max(finite_data)
            if np.isclose(min_val, max_val): min_val -= 0.1; max_val += 0.1
            pad = abs(max_val - min_val) * pad_factor
            if not np.isfinite(pad): pad = abs(max_val)*pad_factor if np.isfinite(max_val) else 1.0
            lim_min, lim_max = min_val - pad, max_val + pad
            if zero_center:
                abs_max = max(abs(lim_min), abs(lim_max))
                if not np.isfinite(abs_max): abs_max = 1.0
                lim_min, lim_max = -abs_max, abs_max
            if lim_max <= lim_min: lim_max = lim_min + 0.1 # Ensure positive range
            return lim_min, lim_max
        # --- End Limit Func ---

        vel_lim = get_lims(all_vels_range, zero_center=True)
        torque_lim = get_lims(torques, zero_center=True)
        kin_err_lim = get_lims(kin_errors, zero_center=True)
        vel_err_lim = get_lims(vel_errors, zero_center=True)

        param_data_for_lims = params_hat
        if self.true_m is not None and params_hat.size > 0:
             param_data_for_lims = np.vstack((params_hat, [[self.true_m, self.true_I]]*params_hat.shape[0]))
        param_lim = get_lims(param_data_for_lims, pad_factor=0.2)
        if param_lim[0] > -0.1: param_lim = (-0.1, param_lim[1])

        dist_lim = get_lims(disturbances, zero_center=True)

        # Apply limits
        axes = [self.velocity_ax, self.torque_ax, self.kin_error_ax, self.vel_error_ax, self.param_ax, self.dist_ax]
        lims = [vel_lim, torque_lim, kin_err_lim, vel_err_lim, param_lim, dist_lim]
        for ax, lim in zip(axes, lims):
             ax.set_xlim(0, max_time)
             ax.set_ylim(lim)

        # Adjust path plot limits dynamically
        path_data = self.sim_data["actual_path"]
        all_x = np.concatenate((self.desired_path[:, 0], path_data[:, 0]))
        all_y = np.concatenate((self.desired_path[:, 1], path_data[:, 1]))
        x_lim = get_lims(all_x)
        y_lim = get_lims(all_y)
        self.path_ax.set_xlim(x_lim)
        self.path_ax.set_ylim(y_lim)

        # Set true parameter lines if available
        if self.true_m is not None:
            self.m_true_line.set_data([0, max_time], [self.true_m, self.true_m])
            self.I_true_line.set_data([0, max_time], [self.true_I, self.true_I])

        self.path_ax.set_title(f"Robot Path Tracking (T=0.00s)")

        return self.artists


    def _animate(self, i):
        """ Animation update function for FuncAnimation """
        # Index for state data (time, path, vels, dist) - Length N+1
        sim_index = min(i * self.anim_step, len(self.sim_data["time"]) - 1)
        # Index for controller/command data (status, torques) - Length N
        controller_index = min(sim_index -1, len(self.controller_status_list) - 1)
        if controller_index < 0: controller_index=0 # Ensure non-negative for slicing

        # State data slices (up to current time sim_index)
        time_data = self.sim_data["time"][:sim_index+1]
        actual_path_data = self.sim_data["actual_path"][:sim_index+1]
        vels_actual_data = self.sim_data["robot_vels"][:sim_index+1]
        dist_data = self.sim_data["disturbances"][:sim_index+1] # Disturbance at step k affects state k+1

        # Command/Controller slices (up to command applied *during* step ending at sim_index)
        # Time axis for these plots: time_data[1:controller_index+2] ? -> time t_1..t_{controller_index+1}
        time_controller = self.sim_data["time"][1:controller_index+2] # Length controller_index+1

        torques_cmd_data = self.sim_data["torques_cmd"][1:controller_index+2] # Torque applied during step k

        # Get controller status up to controller_index
        if controller_index >= 0 and len(self.controller_status_list) > controller_index:
             vels_desired_data = np.array([s['v_d'] for s in self.controller_status_list[:controller_index+1]])
             kin_errors_data = np.array([s['kin_errors'] for s in self.controller_status_list[:controller_index+1]])
             vel_errors_data = np.array([s['eta'] for s in self.controller_status_list[:controller_index+1]])
             params_hat_data = np.array([s['p_hat'] for s in self.controller_status_list[:controller_index+1]])
        else: # Handle first frame(s) where controller_index might be < 0
             time_controller = np.array([])
             vels_desired_data=np.empty((0,2)); kin_errors_data=np.empty((0,3))
             vel_errors_data=np.empty((0,2)); params_hat_data=np.empty((0,2))
             torques_cmd_data=np.empty((0,2)) # Ensure torque data is empty too

        # Get current pose and velocity for markers/arrows
        current_pose = self.sim_data["actual_path"][sim_index]
        current_vel = self.sim_data["robot_vels"][sim_index]
        x, y, theta = current_pose
        v1, omega = current_vel

        # Update Path Plot elements
        self.actual_path_line.set_data(actual_path_data[:, 0], actual_path_data[:, 1])
        self.robot_marker.set_data([x], [y]) # Use lists for set_data with single point

        # Update Arrows - Remove old, create new (simpler than updating properties for non-blit)
        if hasattr(self.robot_arrow, 'axes') and self.robot_arrow.axes: self.robot_arrow.remove()
        arrow_length = max(0.1, np.mean(np.abs(self.path_ax.get_xlim())) * 0.05) # Dynamic arrow size
        self.robot_arrow = patches.Arrow(x, y, arrow_length * np.cos(theta), arrow_length * np.sin(theta),
                                         width=arrow_length * 0.4, color='blue', zorder=5)
        self.path_ax.add_patch(self.robot_arrow)

        if hasattr(self.velocity_arrow, 'axes') and self.velocity_arrow.axes: self.velocity_arrow.remove()
        vel_arrow_scale = arrow_length * 1.5 # Scale velocity arrow relative to orientation
        self.velocity_arrow = patches.Arrow(x, y, v1 * vel_arrow_scale * np.cos(theta), v1 * vel_arrow_scale * np.sin(theta),
                                            width=arrow_length * 0.2, color='lime', alpha=0.7, zorder=4)
        self.path_ax.add_patch(self.velocity_arrow)


        # Update Time Plots safely checking array lengths
        self.v1_actual_line.set_data(time_data, vels_actual_data[:, 0])
        self.omega_actual_line.set_data(time_data, vels_actual_data[:, 1])

        if time_controller.size > 0 and vels_desired_data.shape[0] == time_controller.size:
            self.v1_desired_line.set_data(time_controller, vels_desired_data[:, 0])
            self.omega_desired_line.set_data(time_controller, vels_desired_data[:, 1])
        else: # Clear if no data or mismatch
             self.v1_desired_line.set_data([],[])
             self.omega_desired_line.set_data([],[])

        if time_controller.size > 0 and torques_cmd_data.shape[0] == time_controller.size:
             self.tau_left_line.set_data(time_controller, torques_cmd_data[:, 0])
             self.tau_right_line.set_data(time_controller, torques_cmd_data[:, 1])
        else:
             self.tau_left_line.set_data([],[])
             self.tau_right_line.set_data([],[])

        if time_controller.size > 0 and kin_errors_data.shape[0] == time_controller.size:
            self.error_fwd_line.set_data(time_controller, kin_errors_data[:, 0])
            self.error_lat_line.set_data(time_controller, kin_errors_data[:, 1])
            self.error_theta_line.set_data(time_controller, kin_errors_data[:, 2])
        else:
             self.error_fwd_line.set_data([],[])
             self.error_lat_line.set_data([],[])
             self.error_theta_line.set_data([],[])

        if time_controller.size > 0 and vel_errors_data.shape[0] == time_controller.size:
            self.eta1_line.set_data(time_controller, vel_errors_data[:, 0])
            self.eta2_line.set_data(time_controller, vel_errors_data[:, 1])
        else:
             self.eta1_line.set_data([],[])
             self.eta2_line.set_data([],[])

        if time_controller.size > 0 and params_hat_data.shape[0] == time_controller.size:
            self.m_hat_line.set_data(time_controller, params_hat_data[:, 0])
            self.I_hat_line.set_data(time_controller, params_hat_data[:, 1])
        else:
             self.m_hat_line.set_data([],[])
             self.I_hat_line.set_data([],[])

        # Disturbance plot uses state time axis, slicing disturbance data
        dist_data_slice = dist_data[1:sim_index+1] # Disturbance from step 0 up to step sim_index-1
        time_for_dist = time_data[1:sim_index+1]
        if time_for_dist.size > 0 and dist_data_slice.shape[0] == time_for_dist.size:
             self.dist1_line.set_data(time_for_dist, dist_data_slice[:, 0])
             self.dist2_line.set_data(time_for_dist, dist_data_slice[:, 1])
        else:
             self.dist1_line.set_data([],[])
             self.dist2_line.set_data([],[])


        # Update title
        self.path_ax.set_title(f"Robot Path Tracking (T={time_data[-1]:.2f}s)")

        # Update artist list for blitting if used (but blit=False)
        self.artists = [self.actual_path_line, self.robot_marker,
                 self.v1_actual_line, self.omega_actual_line, self.v1_desired_line, self.omega_desired_line,
                 self.tau_left_line, self.tau_right_line,
                 self.error_fwd_line, self.error_lat_line, self.error_theta_line,
                 self.eta1_line, self.eta2_line,
                 self.m_hat_line, self.I_hat_line,
                 self.m_true_line, self.I_true_line,
                 self.dist1_line, self.dist2_line,
                 self.robot_arrow, self.velocity_arrow] # Include patches

        return self.artists


    def create_animation(self, simulation_data, controller_status_list, interval=50, step=10, true_m=None, true_I=None):
        """ Creates the animation object. """
        self.sim_data = simulation_data
        # Ensure controller_status_list matches number of steps taken
        num_steps = len(simulation_data['time']) - 1
        self.controller_status_list = controller_status_list[:num_steps]
        if len(self.controller_status_list) != num_steps:
             print(f"Warning: Length mismatch in create_animation. Status list {len(self.controller_status_list)}, Steps {num_steps}")


        self.anim_step = step
        self.true_m = true_m
        self.true_I = true_I
        self.path_initialized = False # Reset flag for new animation

        num_sim_points = len(self.sim_data["time"])
        num_frames = (num_sim_points -1) // step + 1 # Ensure enough frames

        # Use blit=False because we are modifying patch artists (Arrows)
        ani = animation.FuncAnimation(self.fig, self._animate, frames=num_frames,
                                      init_func=self._init_animation, blit=False, # Use blit=False
                                      interval=interval, repeat=False)
        return ani


    def plot_final_results(self, simulation_data, controller_status_list, path_type_name="", true_m=None, true_I=None):
        """ Generates static plots of errors, params, etc. vs. time after the simulation. """
        fig_res, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        fig_res.suptitle(f"Adaptive Control Simulation Results - {path_type_name} Path", fontsize=16)

        time_data = simulation_data["time"]
        num_steps = len(controller_status_list)
        if num_steps == 0:
             print("No controller status data to plot.")
             return

        # Ensure time axis length matches number of controller steps
        time_controller = time_data[1:num_steps+1]
        if len(time_controller) != num_steps:
             print(f"Warning: Final plot time axis length ({len(time_controller)}) != controller steps ({num_steps}). Truncating.")
             num_steps = min(len(time_controller), num_steps)
             time_controller = time_controller[:num_steps]
             controller_status_list = controller_status_list[:num_steps] # Truncate status list too

        # Extract controller data safely
        kin_errors = np.array([s['kin_errors'] for s in controller_status_list])
        vel_errors = np.array([s['eta'] for s in controller_status_list])
        params_hat = np.array([s['p_hat'] for s in controller_status_list])

        # State/Command Data (match length to num_steps)
        torques = simulation_data["torques_cmd"][1:num_steps+1]
        disturbances = simulation_data["disturbances"][1:num_steps+1]

        # Plot Kinematic Errors
        axs[0].plot(time_controller, kin_errors[:, 0], 'c-', label="Fwd Err (m)")
        axs[0].plot(time_controller, kin_errors[:, 1], 'y-', label="Lat Err (m)")
        axs[0].plot(time_controller, kin_errors[:, 2], 'k-', label="$\\theta$ Err (rad)")
        axs[0].set_title("Kinematic Errors vs. Time")
        axs[0].set_ylabel("Error")
        axs[0].legend(fontsize='small')
        axs[0].grid(True)

        # Plot Velocity Errors (eta)
        axs[1].plot(time_controller, vel_errors[:, 0], 'r-', label="$\\eta_1 = v_1 - v_{1d}$")
        axs[1].plot(time_controller, vel_errors[:, 1], 'm-', label="$\\eta_2 = \\omega - \\omega_d$")
        axs[1].set_title("Velocity Errors ($\\eta$) vs. Time")
        axs[1].set_ylabel("Vel Error")
        axs[1].legend(fontsize='small')
        axs[1].grid(True)

        # Plot Parameter Estimates
        axs[2].plot(time_controller, params_hat[:, 0], 'b-', label="$\\hat{m}$ (mass)")
        axs[2].plot(time_controller, params_hat[:, 1], 'g-', label="$\\hat{I}$ (inertia)")
        if true_m is not None: axs[2].axhline(true_m, color='b', linestyle='--', alpha=0.7, label="$m$ True")
        if true_I is not None: axs[2].axhline(true_I, color='g', linestyle='--', alpha=0.7, label="$I$ True")
        axs[2].set_title("Parameter Estimates vs. Time")
        axs[2].set_ylabel("Value")
        axs[2].legend(fontsize='small')
        axs[2].grid(True)

        # Plot Torques
        if torques.shape[0] == num_steps:
             axs[3].plot(time_controller, torques[:, 0], 'b-', label="$\\tau_L$ Left")
             axs[3].plot(time_controller, torques[:, 1], 'g-', label="$\\tau_R$ Right")
        axs[3].set_title("Control Torques vs. Time")
        axs[3].set_ylabel("Torque (Nm)")
        axs[3].legend(fontsize='small')
        axs[3].grid(True)

        # Plot Disturbances
        if disturbances.shape[0] == num_steps:
             axs[4].plot(time_controller, disturbances[:, 0], 'r-', label="$\\tau_{d1}$ ($v_1$ eq)")
             axs[4].plot(time_controller, disturbances[:, 1], 'm-', label="$\\tau_{d2}$ ($\\omega$ eq)")
        axs[4].set_title("Actual Disturbances vs. Time")
        axs[4].set_xlabel("Time (s)")
        axs[4].set_ylabel("Disturbance")
        axs[4].legend(fontsize='small')
        axs[4].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout