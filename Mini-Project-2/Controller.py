import numpy as np
import time

class Controller:
    def __init__(self):
        """ Base controller class. """
        pass

    def compute_control(self, robot_state, predefined_path):
        """ Compute control commands. Must be implemented by subclasses. """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _safe_sinc(self, x):
        """ Calculates sin(x)/x, handling x=0. """
        if np.abs(x) < 1e-6:
            # Use Taylor expansion: sin(x)/x = (x - x^3/6 + ...)/x = 1 - x^2/6 + ...
            return 1.0 - x**2 / 6.0
        else:
            return np.sin(x) / x

# --- LyapunovKinematicController ---
class LyapunovKinematicController(Controller):
    def __init__(self, k_forward=1.0, k_theta=2.0, k_lateral_gain_factor=1.0,
                 v_ref=1.0, omega_max=np.pi, lookahead_dist=0.5):
        """
        Lyapunov-based kinematic controller to generate desired v1_d, omega_d.
        :param k_forward: Gain for forward error correction.
        :param k_theta: Gain for orientation error correction.
        :param k_lateral_gain_factor: Factor scaling the lateral error correction term.
        :param v_ref: Reference forward velocity along the path.
        :param omega_max: Maximum allowable angular velocity command.
        :param lookahead_dist: Lookahead distance to find target point.
        """
        super().__init__()
        self.kf = k_forward
        self.ktt = k_theta
        self.k_lat_factor = k_lateral_gain_factor
        self.v_ref = v_ref
        self.omega_max = omega_max
        self.lookahead_dist = lookahead_dist
        if lookahead_dist <= 0:
             print("Warning: lookahead_dist should be positive. Setting to 0.1.")
             self.lookahead_dist = 0.1
        self.closest_index = 0
        # Store previous desired velocity state for derivative calculation if needed by adaptive controller
        self.prev_v1d = 0.0
        self.prev_omegad = 0.0
        self.finished_flag = False # Internal flag

    def find_target_point(self, robot_pos, path):
        """ Find the lookahead point on the path """
        current_pos = np.array(robot_pos)
        distances = np.linalg.norm(path - current_pos, axis=1)

        # Find closest point index (limited search for efficiency)
        search_radius = 20
        start_idx = max(0, self.closest_index - search_radius)
        end_idx = min(len(path), self.closest_index + search_radius + 1)
        path_segment = path[start_idx:end_idx]
        if path_segment.shape[0] == 0: # Path is shorter than search radius start
             if len(path) > 0:
                 self.closest_index = np.argmin(distances) # Search whole path
             else:
                 return None, None # No path

        else:
             relative_closest_idx = np.argmin(np.linalg.norm(path_segment - current_pos, axis=1))
             self.closest_index = start_idx + relative_closest_idx

        # Find lookahead point by searching forward from closest point
        target_idx = self.closest_index
        found_lookahead = False
        while target_idx < len(path) - 1:
            dist = np.linalg.norm(path[target_idx] - current_pos)
            if dist >= self.lookahead_dist:
                found_lookahead = True
                break
            target_idx += 1

        # If lookahead search reached end, or if robot is close to the final point
        dist_to_end_point = np.linalg.norm(path[-1] - current_pos)

        if not found_lookahead and dist_to_end_point < self.lookahead_dist * 1.5:
            # Target the last point if near the end
            target_point = path[-1]
            target_idx = len(path) - 1 # Ensure index points to last element
            if dist_to_end_point < 0.15: # Close enough criteria to signal finish
                self.finished_flag = True
                # print("Kinematic controller: Reached end target.")
                # Return None for target, let compute_desired handle finished flag
        else:
            # Otherwise, use the found target_idx (could be last point if loop finished)
            target_point = path[target_idx]


        # Calculate path tangent at target point (approximate)
        # Use segment ending at target point for tangent, unless at start/end or duplicate points
        if target_idx > 0:
            vec = path[target_idx] - path[target_idx - 1]
        elif len(path) > 1: # At the very start (target_idx is 0)
            vec = path[1] - path[0]
        else: # Path has only one point
            vec = np.array([1.0, 0.0]) # Assume horizontal tangent

        # Handle zero-length vector case (e.g., duplicate points)
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-6:
            # Try segment *after* target point if possible
            if target_idx + 1 < len(path):
                 vec = path[target_idx+1] - path[target_idx]
                 vec_norm = np.linalg.norm(vec)
            # If still zero or couldn't look ahead, default to horizontal or keep previous?
            if vec_norm < 1e-6:
                vec = np.array([1.0, 0.0]) # Default to horizontal

        theta_path_tangent = np.arctan2(vec[1], vec[0])

        return target_point, theta_path_tangent


    def compute_desired_velocities(self, robot_state, predefined_path):
        """
        Compute desired kinematic velocities (v1_d, omega_d).
        :param robot_state: Current state (x, y, theta, v1, omega).
        :param predefined_path: Target path (N, 2).
        :return: Tuple (v1_d, omega_d, errors=(e_fwd, e_lat, e_theta), finished)
        """
        # Reset finished flag at start of computation
        # self.finished_flag = False # Resetting here might cause issues if called multiple times per step

        if self.finished_flag: # If already flagged as finished, return zero velocities
             return 0.0, 0.0, (0.0, 0.0, 0.0), True

        x, y, theta, _, _ = robot_state # Unpack pose
        current_pos = (x, y)

        if predefined_path is None or len(predefined_path) == 0:
            return 0.0, 0.0, (0.0, 0.0, 0.0), True # No path

        target_point, theta_path_tangent = self.find_target_point(current_pos, predefined_path)

        # Check finished_flag which might be set by find_target_point
        if self.finished_flag:
             # Smoothly decelerate if finished flag is set
             self.prev_v1d = 0.0 # Store zero for next step's derivative
             self.prev_omegad = 0.0
             return 0.0, 0.0, (0.0, 0.0, 0.0), True


        x_d, y_d = target_point

        # --- Error Calculation ---
        error_x = x_d - x
        error_y = y_d - y

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        error_forward = error_x * cos_t + error_y * sin_t
        error_lateral = -error_x * sin_t + error_y * cos_t
        error_theta = (theta_path_tangent - theta + np.pi) % (2 * np.pi) - np.pi

        # --- Kinematic Control Law ---
        v1_d = self.v_ref * np.cos(error_theta) + self.kf * error_forward
        k_lat = self.ktt * self.k_lat_factor
        omega_d = k_lat * self.v_ref * self._safe_sinc(error_theta) * error_lateral + self.ktt * error_theta

        # Limit velocities
        omega_d = np.clip(omega_d, -self.omega_max, self.omega_max)
        # Optional: Limit linear velocity if needed (e.g., based on curvature or fixed max)
        # max_linear_vel = 1.5 * self.v_ref
        # v1_d = np.clip(v1_d, 0, max_linear_vel) # Allow only forward motion? Depends on strategy

        # Store previous for derivative calculation if needed by adaptive controller
        self.prev_v1d = v1_d
        self.prev_omegad = omega_d

        errors = (error_forward, error_lateral, error_theta)
        return v1_d, omega_d, errors, self.finished_flag


# --- Adaptive Dynamic Controller ---
class AdaptiveDynamicController(Controller):
    def __init__(self, dt, wheel_radius, wheel_width,
                 kinematic_controller: LyapunovKinematicController, # Takes a kinematic controller instance
                 initial_p_hat=np.array([1.0, 0.1]), # Initial estimate for [m, I]
                 gamma_p=np.diag([1.0, 0.1]),       # Adaptation gain matrix for p_hat
                 kd=np.diag([10.0, 5.0]),             # Velocity error feedback gain K_d
                 disturbance_bound=0.5,               # Assumed disturbance bound d_B for robust term
                 use_robust_term=True,
                 # Parameter bounds (optional but recommended)
                 min_params = np.array([0.1, 0.01]), # Min m, min I
                 max_params = np.array([100.0, 50.0]) # Max m, max I
                 ):
        """
        Adaptive controller based on paper, simplified.
        Takes torque commands to track desired velocities from a kinematic controller.
        :param dt: Simulation time step.
        :param wheel_radius: Radius of the wheels (r).
        :param wheel_width: Distance between the wheels (W).
        :param kinematic_controller: Instance of a kinematic controller providing v_d.
        :param initial_p_hat: Initial estimate vector for parameters [m, I].
        :param gamma_p: Adaptation gain matrix (positive definite) for p_hat.
        :param kd: Gain matrix (positive definite) for velocity error feedback.
        :param disturbance_bound: Assumed upper bound d_B for robust control term.
        :param use_robust_term: Boolean flag to enable/disable the sgn(eta) term.
        :param min_params: Optional minimum bounds for parameter estimates [m, I].
        :param max_params: Optional maximum bounds for parameter estimates [m, I].
        """
        super().__init__()
        self.dt = dt
        if dt <= 0: raise ValueError("Time step dt must be positive.")
        self.kinematic_controller = kinematic_controller
        self.p_hat = np.array(initial_p_hat, dtype=float) # Estimate [m, I]
        self.Gamma_p = np.array(gamma_p, dtype=float)
        self.Kd = np.array(kd, dtype=float)
        self.dB = disturbance_bound # Corresponds to d_B in paper
        self.use_robust_term = use_robust_term
        self.min_params = np.array(min_params)
        self.max_params = np.array(max_params)
        if np.any(self.min_params <=0): print("Warning: min_params should be positive.")
        if np.any(self.max_params <= self.min_params): raise ValueError("max_params must be greater than min_params")

        # Precompute B2 inverse matrix
        r = wheel_radius
        W = wheel_width
        if W == 0: raise ValueError("Wheel width cannot be zero")
        self.B2_inv = (r / W) * np.array([
            [W / 2.0, 1.0],
            [W / 2.0, -1.0]
        ])

        # Store previous desired velocities for derivative estimate
        self.prev_v1d_kin = 0.0
        self.prev_omegad_kin = 0.0

        # State variables for filtering vd_dot (optional, uncomment to use)
        # self.prev_v1d_filtered = 0.0
        # self.prev_omegad_filtered = 0.0
        # self.prev_filtered_v1d_for_deriv = 0.0
        # self.prev_filtered_omegad_for_deriv = 0.0

        self.first_run = True # Flag to handle initialization


    def compute_control(self, robot_state, predefined_path):
        """
        Compute adaptive torque commands.
        :param robot_state: Current full state (x, y, theta, v1, omega).
        :param predefined_path: Target path (N, 2).
        :return: Tuple (tau_left, tau_right, status)
                 status = {'eta': eta, 'p_hat': p_hat, 'kin_errors': kin_errors, 'v_d': v_d}
        """
        x, y, theta, v1, omega = robot_state
        v_actual = np.array([v1, omega])

        # 1. Get desired kinematic velocities (v_d) from the kinematic controller
        v1d, omegad, kin_errors, finished = self.kinematic_controller.compute_desired_velocities(robot_state, predefined_path)
        v_d = np.array([v1d, omegad])

        # Initialize derivative state on first run
        if self.first_run:
            self.prev_v1d_kin = v1d
            self.prev_omegad_kin = omegad
            # Initialize filter states if using filter:
            # self.prev_v1d_filtered = v1d
            # self.prev_omegad_filtered = omegad
            # self.prev_filtered_v1d_for_deriv = v1d
            # self.prev_filtered_omegad_for_deriv = omegad
            self.first_run = False

        # Handle finished state from kinematic controller
        if finished:
             # Calculate final status before returning zero torque
             eta = v_actual - v_d # Use last computed vd
             status = {'eta': eta, 'p_hat': self.p_hat.copy(), 'kin_errors': kin_errors, 'v_d': v_d}
             return 0.0, 0.0, status

        # 2. Estimate derivative of desired velocities (simple numerical differentiation)
        # --- Option A: Original Noisy Derivative ---
        v1d_dot = (v1d - self.prev_v1d_kin) / self.dt
        omegad_dot = (omegad - self.prev_omegad_kin) / self.dt
        # Update previous values for next step's derivative calc
        self.prev_v1d_kin = v1d
        self.prev_omegad_kin = omegad
        # --- End Option A ---

        # --- Option B: Filtered Derivative Placeholder (Uncomment block & init vars to use) ---
        # # Filter vd before differentiation (Simple Exponential Moving Average example)
        # alpha_filter = 0.3 # Smoothing factor (0 < alpha <= 1, smaller is smoother)
        # filtered_v1d = alpha_filter * v1d + (1 - alpha_filter) * self.prev_v1d_filtered
        # filtered_omegad = alpha_filter * omegad + (1 - alpha_filter) * self.prev_omegad_filtered
        # # Estimate derivative using filtered values
        # v1d_dot = (filtered_v1d - self.prev_filtered_v1d_for_deriv) / self.dt
        # omegad_dot = (filtered_omegad - self.prev_filtered_omegad_for_deriv) / self.dt
        # # Update stored filtered values for next step
        # self.prev_v1d_filtered = filtered_v1d
        # self.prev_omegad_filtered = filtered_omegad
        # self.prev_filtered_v1d_for_deriv = filtered_v1d
        # self.prev_filtered_omegad_for_deriv = filtered_omegad
        # --- End Option B ---

        vd_dot = np.array([v1d_dot, omegad_dot])

        # 3. Calculate velocity error (eta = v_actual - v_desired)
        eta = v_actual - v_d

        # 4. Calculate Regressor Matrix Yc
        # Yc*p = M2*vd_dot => Yc = diag(vd_dot) for diagonal M2
        Yc = np.diag(vd_dot)

        # 5. Update Parameter Estimates (p_hat for [m, I]) using adaptive law
        p_hat_dot = -self.Gamma_p @ Yc.T @ eta

        # Euler integration for p_hat
        self.p_hat += p_hat_dot * self.dt

        # Apply bounds to parameter estimates
        self.p_hat = np.maximum(self.p_hat, self.min_params)
        self.p_hat = np.minimum(self.p_hat, self.max_params)

        # 6. Calculate Robust Control Term (u_robust = dB * sgn(eta))
        if self.use_robust_term:
            epsilon = 0.05 # Boundary layer thickness
            tanh_arg = eta / epsilon
            tanh_arg = np.clip(tanh_arg, -10, 10) # Prevent overflow
            sgn_eta_smoothed = np.tanh(tanh_arg)
            u_robust = self.dB * sgn_eta_smoothed
        else:
            u_robust = np.zeros_like(eta)

        # 7. Calculate Control Torque Tau
        # B2*tau = Yc*p_hat - Kd*eta - u_robust
        term1 = Yc @ self.p_hat
        term2 = self.Kd @ eta
        term3 = u_robust

        tau_bar = term1 - term2 - term3
        tau_cmd = self.B2_inv @ tau_bar # tau_cmd = [tau_r, tau_l]^T

        tau_right, tau_left = tau_cmd

        # Optional: Add torque limits if needed
        # max_torque = 10.0
        # tau_left = np.clip(tau_left, -max_torque, max_torque)
        # tau_right = np.clip(tau_right, -max_torque, max_torque)

        # Store intermediate values for analysis/plotting
        status = {'eta': eta, 'p_hat': self.p_hat.copy(), 'kin_errors': kin_errors, 'v_d': v_d}

        return tau_left, tau_right, status