import numpy as np
import matplotlib.pyplot as plt
import time  # Ensure time is imported
from Simulation import Simulation
from Controller import LyapunovKinematicController, AdaptiveDynamicController
from Visualizer import Visualizer # Import the Visualizer

# Dictionary mapping type index to name for filenames/titles
PATH_TYPES = {
    1: "Circle", 2: "Ellipse", 3: "Spiral", 4: "Line", 5: "Lemniscate",
    6: "SineWave", 7: "Heart", 8: "SquareWave", 9: "Parabola", 10: "Complex"
}

def generate_path(path_type_index=2):
    num_points = 150 # Number of points in the path
    path_type = str(path_type_index)

    if path_type == '1': # Circle
        radius = 5
        t = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
    elif path_type == '2': # Ellipse
        rx, ry = 6, 4
        t = np.linspace(0, 2 * np.pi, num_points)
        x = rx * np.cos(t)
        y = ry * np.sin(t)
    elif path_type == '3': # Spiral
        a = 0.5 # Controls tightness
        t = np.linspace(0, 6 * np.pi, num_points) # More turns
        x = a * t * np.cos(t)
        y = a * t * np.sin(t)
    elif path_type == '4': # Line
        x = np.linspace(0, 10, num_points)
        y = 0.5 * x + 1 # Example: y = 0.5x + 1
    elif path_type == '5': # Lemniscate (Figure-eight)
        scale = 5
        t = np.linspace(0, 2 * np.pi, num_points)
        x = scale * np.cos(t) / (1 + np.sin(t)**2)
        y = scale * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    elif path_type == '6': # Sine wave
        amp = 2
        freq = 1.5
        x = np.linspace(0, 4 * np.pi / freq, num_points)
        y = amp * np.sin(freq * x)
    elif path_type == '7': # Heart shape
        t = np.linspace(0, 2 * np.pi, num_points)
        x = 16 * np.sin(t)**3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        x = x/3 # Scale down
        y = y/3 # Scale down
    elif path_type == '8': # Square wave-like path (smoothed slightly)
        x_seg = np.linspace(0, 10, 6) # Define corners
        y_seg = [0, 2, 2, -2, -2, 0]
        x, y = [], []
        points_per_seg = num_points // (len(x_seg)-1) if len(x_seg) > 1 else num_points
        for i in range(len(x_seg)-1):
             x_pts = np.linspace(x_seg[i], x_seg[i+1], points_per_seg)
             y_pts = np.linspace(y_seg[i], y_seg[i+1], points_per_seg)
             x.extend(x_pts[:-1])
             y.extend(y_pts[:-1])
        x.append(x_seg[-1])
        y.append(y_seg[-1])
        x = np.array(x)
        y = np.array(y)

    elif path_type == '9': # Parabola
        x = np.linspace(-5, 5, num_points)
        y = 0.3*x**2
    elif path_type == '10': # Complex
        t = np.linspace(0, 5 * np.pi, num_points)
        x = (t/3) * np.cos(t) + 3 * np.sin(1.5 * t)
        y = (t/3) * np.sin(t) + 2 * np.cos(t/2)
    else:
        raise ValueError(f"Invalid path type index: {path_type_index}. Choose from {list(PATH_TYPES.keys())}")

    return np.column_stack((x, y))


def main():
    # --- Simulation Parameters ---
    selected_path_type = 10 # Choose path type index (e.g., 6 for SineWave, 10 for Complex)
    path_name = PATH_TYPES.get(selected_path_type, "Custom")
    print(f"Generating path: {path_name} (Type {selected_path_type})")

    desired_path = generate_path(selected_path_type)
    if desired_path.shape[0] < 2:
         print("Error: Generated path has less than 2 points.")
         return

    # Initial robot state (pose and velocity)
    start_dx = desired_path[1, 0] - desired_path[0, 0]
    start_dy = desired_path[1, 1] - desired_path[0, 1]
    initial_theta = np.arctan2(start_dy, start_dx)
    initial_offset = 0.3 # Distance offset from start
    initial_pose = np.array([
        desired_path[0, 0] - initial_offset * np.sin(initial_theta),
        desired_path[0, 1] + initial_offset * np.cos(initial_theta),
        initial_theta + 0.2 # Small angle offset
    ])
    initial_velocity = np.array([0.0, 0.0]) # Start from rest

    dt = 0.02           # Simulation time step (s)
    wheel_radius = 0.05 # m (r)
    wheel_width = 0.3   # m (W)
    max_steps = 3000    # Maximum simulation steps (increase for complex paths)

    # --- Robot Dynamic Parameters (Actual Values for Simulation) ---
    robot_mass = 15.0      # kg (m)
    robot_inertia = 1.2    # kg*m^2 (I)
    # disturbance_level = 0.0 # Set to 0 for no disturbance initially
    disturbance_level = 0.5 # Max disturbance torque in simulation (Nm)

    # --- Kinematic Controller Parameters ---
    kin_k_forward = 1.8
    kin_k_theta = 4.5         # Sensitive gain, adjust carefully
    kin_k_lateral_factor = 1.0
    kin_v_ref = 0.8           # Target speed for kinematic controller
    kin_omega_max = np.pi * 1.5 # Max desired angular velocity
    kin_lookahead = 0.4       # Lookahead distance

    # --- Adaptive Controller Parameters ---
    # Initial estimates for [m, I] - Start closer for stability?
    initial_p_hat = np.array([robot_mass * 0.8, robot_inertia * 1.5])
    # Adaptation gains for p_hat = [m_hat, I_hat] - Lower these significantly!
    gamma_p = np.diag([0.05, 0.005]) # << ADJUST THESE >>
    # Velocity error feedback gains Kd = diag(kd1, kd2) - Proportional to initial estimates?
    kd1_factor = 5.0 # Factor multiplied by mass estimate
    kd2_factor = 8.0 # Factor multiplied by inertia estimate
    kd = np.diag([initial_p_hat[0] * kd1_factor, initial_p_hat[1] * kd2_factor])
    # Assumed disturbance bound for controller's robust term
    controller_dB = disturbance_level * 1.2 # Controller assumes slightly higher bound
    use_robust = True # Enable robust term in controller? Try False first.

    # Parameter estimate bounds (min_m, min_I), (max_m, max_I)
    min_params=np.array([1.0, 0.1])
    max_params=np.array([50.0, 10.0])


    # --- Initialization ---
    simulation = Simulation(dt, desired_path, wheel_radius, wheel_width,
                            m=robot_mass, I=robot_inertia, dB=disturbance_level,
                            initial_pose=initial_pose, initial_velocity=initial_velocity)

    kinematic_controller = LyapunovKinematicController(
        k_forward=kin_k_forward, k_theta=kin_k_theta,
        k_lateral_gain_factor=kin_k_lateral_factor,
        v_ref=kin_v_ref, omega_max=kin_omega_max, lookahead_dist=kin_lookahead
    )

    adaptive_controller = AdaptiveDynamicController(
        dt=dt, wheel_radius=wheel_radius, wheel_width=wheel_width,
        kinematic_controller=kinematic_controller,
        initial_p_hat=initial_p_hat,
        gamma_p=gamma_p,
        kd=kd,
        disturbance_bound=controller_dB,
        use_robust_term=use_robust,
        min_params=min_params,
        max_params=max_params
    )

    # --- Data Storage ---
    controller_status_list = [] # To store eta, p_hat, etc.

    # --- Simulation Loop ---
    print("Starting simulation...")
    start_time_sim = time.time()
    controller_finished = False
    for step in range(max_steps):
        current_state = simulation.get_robot_state()

        # Compute control torques only if not finished
        if not controller_finished:
             try:
                 tau_left_cmd, tau_right_cmd, status = adaptive_controller.compute_control(current_state, desired_path)
                 controller_status_list.append(status) # Store status only if compute_control succeeds

                 # Check if kinematic controller signaled finish via the flag in its instance
                 if adaptive_controller.kinematic_controller.finished_flag:
                       print(f"Controller signaled finish at step {step}.")
                       controller_finished = True
                       # Command zero torque after finishing
                       tau_left_cmd, tau_right_cmd = 0.0, 0.0

             except Exception as e:
                  print(f"Error in controller at step {step}: {e}")
                  print(f"Current state: {current_state}")
                  # Option: Stop simulation or try to continue with zero torque
                  tau_left_cmd, tau_right_cmd = 0.0, 0.0
                  # Optionally break the loop
                  # break
        else:
             # Keep applying zero torque if finished
             tau_left_cmd, tau_right_cmd = 0.0, 0.0

        # Execute command in simulation
        try:
             simulation.execute_cmd(tau_left_cmd, tau_right_cmd)
        except Exception as e:
             print(f"Error in simulation step {step}: {e}")
             print(f"Commanded torques: L={tau_left_cmd}, R={tau_right_cmd}")
             break # Stop simulation if physics fail

        # Optional: Print progress
        if (step + 1) % 200 == 0 and controller_status_list:
             last_status = controller_status_list[-1]
             print(f"Step: {step+1}/{max_steps}, Time: {simulation.time_stamps[-1]:.2f}s, "
                   f"Pos:({current_state[0]:.2f},{current_state[1]:.2f}), "
                   f"Vels:({current_state[3]:.2f},{current_state[4]:.2f}), "
                   f"p_hat:[{last_status['p_hat'][0]:.2f}, {last_status['p_hat'][1]:.2f}]")

        # Break loop if finished
        if controller_finished and step > adaptive_controller.kinematic_controller.closest_index + 5: # Allow a few steps after finish signal
             print(f"Simulation loop ending early at step {step} due to controller finish.")
             break

    else: # Runs if loop completes without break
        print(f"Simulation finished after {max_steps} steps.")
    end_time_sim = time.time()
    print(f"Simulation duration: {end_time_sim - start_time_sim:.2f} seconds")


    # --- Post-Simulation Processing ---
    simulation_data = simulation.get_simulation_data()
    num_sim_data_points = len(simulation_data['time'])
    num_controller_points = len(controller_status_list)
    print(f"Sim data points: {num_sim_data_points}, Controller status points: {num_controller_points}")
    # Adjust controller list length if simulation ended early
    if num_controller_points > num_sim_data_points -1 :
         print("Adjusting controller status list length...")
         controller_status_list = controller_status_list[:num_sim_data_points-1]


    # --- Visualization and Animation ---
    print("Creating visualization and animation...")
    visualizer = Visualizer(desired_path)

    # Animation settings
    anim_interval = 50  # ms between frames
    anim_step = 3       # Simulation steps per animation frame

    # Create animation
    ani = visualizer.create_animation(simulation_data, controller_status_list,
                                      interval=anim_interval, step=anim_step,
                                      true_m=robot_mass, true_I=robot_inertia)

    # Save animation
    gif_filename = f'robot_adaptive_animation_{path_name.replace(" ", "")}.gif'
    print(f"Saving animation to {gif_filename}...")
    try:
        time_per_frame = anim_step * dt
        fps = max(1, int(1.0 / time_per_frame)) if time_per_frame > 0 else 10
        print(f"Calculated FPS for saving: {fps}")
        # Increase writer patience if saving takes long
        ani.save(gif_filename, writer='pillow', fps=fps, progress_callback=lambda i, n: print(f'Saving frame {i+1}/{n}') if (i+1)%50==0 or i+1==n else None)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure you have 'pillow' installed (`pip install pillow`) and potentially ffmpeg if saving as mp4.")

    # Plot final static results
    print("Plotting final results...")
    visualizer.plot_final_results(simulation_data, controller_status_list, path_name,
                                 true_m=robot_mass, true_I=robot_inertia)

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()