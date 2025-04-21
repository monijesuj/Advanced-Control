import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from .pendulum import Pendulum # Changed to relative import

class Plotter:
    def __init__(self, time_vector: np.ndarray | None, state_history: np.ndarray | None, control_history: np.ndarray | None, system: Pendulum | None):
        """
        Initializes the Plotter.

        Args:
            time_vector: Array of time points. Can be None if using plot_multiple_phase_portraits.
            state_history: Array of state vectors over time. Can be None if using plot_multiple_phase_portraits.
            control_history: Array of [control_value, controller_index] pairs over time. Can be None if using plot_multiple_phase_portraits.
            system: The Pendulum system instance. Can be None if using plot_multiple_phase_portraits.
        """
        self.t = time_vector
        self.states = state_history
        self.system = system # Store the system object
        self.controls = control_history # Store the 2D control history [value, index]
        self.E_kin_history = None
        self.E_pot_history = None
        self.E_tot_history = None
        self.E_des = None

        # Perform checks and calculations only if data for a single run is provided
        if self.t is not None and self.states is not None and self.controls is not None and self.system is not None:
            # Ensure control history has NaN padding for the last point if needed for plotting
            # Assumes control_history comes from Simulation.run which has length num_steps (len(t)-1)
            if self.controls.shape[0] == len(self.t) - 1:
                # Append a row of NaNs if control has one less entry than time
                nan_row = np.full((1, self.controls.shape[1]), np.nan)
                self.controls = np.vstack([self.controls, nan_row])
            elif self.controls.shape[0] == len(self.t):
                # Keep as is if lengths match
                pass
            else:
                 raise ValueError(f"Control history shape {self.controls.shape} mismatch with time vector length {len(self.t)}.")

            if self.states.shape[0] != len(self.t):
                raise ValueError("State history must have the same length as the time vector.")
            if self.states.shape[1] != 2:
                raise ValueError("State history must have 2 columns (theta, theta_dot).")

            # Calculate energy history
            self.E_kin_history = np.array([self.system.get_kinetic_energy(s) for s in self.states])
            self.E_pot_history = np.array([self.system.get_potential_energy(s) for s in self.states])
            self.E_tot_history = self.E_kin_history + self.E_pot_history
            # Check if the system has a desired energy (some controllers might not)
            if hasattr(self.system, 'get_desired_energy'):
                self.E_des = self.system.get_desired_energy()

    def plot_results(self):
        """Plots the state variables, control input, energies, and phase portrait using GridSpec."""
        if self.t is None or self.states is None or self.controls is None:
            raise ValueError("Cannot plot results: missing time, state or control data.")

        # Extract state variables
        theta = self.states[:, 0]  # Angle (rad)
        theta_dot = self.states[:, 1]  # Angular velocity (rad/s)

        # Create a figure with GridSpec for complex layout
        fig = plt.figure(figsize=(14, 15), constrained_layout=True)  # Use constrained_layout for better spacing
        gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[2, 1, 1, 1, 1.2])

        # Create subplot axes
        ax_state = fig.add_subplot(gs[0, 0])   # Row 0, Col 0
        ax_control = fig.add_subplot(gs[0, 1])  # Row 0, Col 1
        ax_phase = fig.add_subplot(gs[1:4, :])  # Rows 1-3, Cols 0-1 (3x2 block)
        ax_energy = fig.add_subplot(gs[4, :])     # Row 4, All Cols

        # --- Plotting Data --- #
        # 1. State Variables (Top Left)
        ax_state.plot(self.t, theta, label='Theta (rad)', color='blue')
        ax_state.plot(self.t, theta_dot, label='Theta_dot (rad/s)', color='orange', linestyle='--')
        ax_state.set_title('State Variables')
        ax_state.set_ylabel('Value')
        ax_state.grid(True)
        ax_state.legend()

        # 2. Control Input (Top Right)
        # Use only the control values (first column) for plotting
        control_values = self.controls[:, 0]
        time_for_control = self.t[:-1] if np.isnan(control_values[-1]) else self.t
        controls_to_plot = control_values[:len(time_for_control)]
        ax_control.plot(time_for_control, controls_to_plot, label='Control Torque (Nm)', color='green')

        ax_control.set_title('Control Input')
        ax_control.set_ylabel('Torque (Nm)')
        ax_control.grid(True)
        ax_control.legend(loc='upper left') # Only one legend needed now

        # 3. Phase Portrait (Middle Full Width)
        scatter = ax_phase.scatter(theta, theta_dot, c=self.t, cmap='cool', s=10, label='Phase Trajectory', alpha=0.6)
        try:
             ax_phase.scatter(theta[0], theta_dot[0], color='red', s=150, label='Start', zorder=5, alpha=0.7)
             ax_phase.scatter(theta[-1], theta_dot[-1], color='black', s=150, label='End', zorder=5, alpha=0.7)
        except IndexError:
             print("Warning: Could not plot start/end points (short simulation?).")
        ax_phase.set_title('Phase Portrait (Theta vs Theta_dot)')
        ax_phase.set_xlabel(r'$\theta$ (rad)', fontsize=12)
        ax_phase.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
        ax_phase.grid(True)
        
        # Автоматически определяем диапазон осей на основе фактических данных
        theta_range = max(theta) - min(theta)
        theta_dot_range = max(theta_dot) - min(theta_dot)
        
        # Добавляем небольшой отступ (10%)
        theta_padding = theta_range * 0.1
        theta_dot_padding = theta_dot_range * 0.1
        
        ax_phase.set_xlim(min(theta) - theta_padding, max(theta) + theta_padding)
        ax_phase.set_ylim(min(theta_dot) - theta_dot_padding, max(theta_dot) + theta_dot_padding)
        
        # Добавляем цветовую шкалу для времени
        cbar = plt.colorbar(scatter, ax=ax_phase)
        cbar.set_label('Time (s)')
        
        # Маркируем особые точки на фазовой плоскости
        ax_phase.scatter(0, 0, marker='x', color='red', s=250, label='Bottom Position (Stable)', zorder=6, alpha=0.8)
        ax_phase.scatter(np.pi, 0, marker='o', color='green', s=250, label='Top Position (Unstable Target)', zorder=6, alpha=0.8)

        ax_phase.legend(loc='best')
        
        # 4. Energy Plot (Bottom Full Width)
        if self.E_kin_history is not None:
            ax_energy.plot(self.t, self.E_kin_history, label='Kinetic Energy', color='blue', alpha=0.7)
            ax_energy.plot(self.t, self.E_pot_history, label='Potential Energy', color='green', alpha=0.7)
            ax_energy.plot(self.t, self.E_tot_history, label='Total Energy', color='red', linewidth=2)
            
            if self.E_des is not None:
                ax_energy.axhline(y=self.E_des, color='black', linestyle='--', label='Desired Energy')
            
            ax_energy.set_title('Energy Components')
            ax_energy.set_xlabel('Time (s)')
            ax_energy.set_ylabel('Energy (Joules)')
            ax_energy.grid(True)
            ax_energy.legend(loc='upper right')

        plt.show()

    def plot_multiple_phase_portraits(self, simulation_results_list: list[tuple[np.ndarray, np.ndarray]],
                                      equilibrium_point: np.ndarray = np.array([0.0, 0.0]),
                                      k_coeffs: tuple[float, float] | None = None,
                                      eigenvalues: tuple[complex, complex] | None = None,
                                      title: str = "Phase Portraits",
                                      plot_range: tuple[float, float, float, float] | None = None):
        """
        Plots multiple phase portraits on the same figure.

        Args:
            simulation_results_list: A list of tuples, where each tuple contains
                                     (time_vector, state_history). state_history
                                     should have shape (num_steps+1, 2).
            equilibrium_point: The equilibrium point [theta, theta_dot] to plot.
            k_coeffs: Optional tuple (k1, k2) for title annotation.
            eigenvalues: Optional tuple (lambda1, lambda2) for title annotation.
            title: The title for the plot.
            plot_range: Optional tuple (xmin, xmax, ymin, ymax) for axis limits.
        """
        # Увеличим размер фигуры для лучшего отображения
        fig, ax = plt.subplots(figsize=(12, 10))

        # Setup colormap (similar to cool -> magenta)
        cmap = cm.cool
        # Use the first time vector for normalization (assuming time is the same)
        t_norm_ref = simulation_results_list[0][0]
        norm = mcolors.Normalize(vmin=t_norm_ref.min(), vmax=t_norm_ref.max())

        print(f"Plotting {len(simulation_results_list)} trajectories...")

        all_theta = []
        all_theta_dot = []

        for i, (t, states) in enumerate(simulation_results_list):
            theta = states[:, 0]
            theta_dot = states[:, 1]
            all_theta.extend(theta)
            all_theta_dot.extend(theta_dot)

            points = np.array([theta, theta_dot]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create LineCollection
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            # Set segment colors based on time (use midpoint time of segment)
            segment_times = (t[:-1] + t[1:]) / 2
            lc.set_array(segment_times)
            lc.set_linewidth(1.5) # Slightly thinner, as in the example
            line = ax.add_collection(lc)

            # Initial point (blue circle) - увеличен размер для лучшей видимости
            ax.plot(theta[0], theta_dot[0], 'o', color='blue', markersize=8, label='Initial Point' if i == 0 else "")
            # Final point (red cross) - Add this line
            ax.plot(theta[-1], theta_dot[-1], 'x', color='red', markersize=8, label='Final Point' if i == 0 else "")

        # Equilibrium point (red circle with outline) - увеличен размер
        ax.plot(equilibrium_point[0], equilibrium_point[1], 'o', color='firebrick', markersize=10, label='Equilibrium Point')
        ax.plot(equilibrium_point[0], equilibrium_point[1], 'o', mec='black', mfc='none', markersize=10) # Outline

        # Add vertical line at theta = pi with correct raw string label
        ax.axvline(np.pi, color='red', linestyle=':', linewidth=2.0, label=r'$\theta = \pi$')

        # Add axes lines crossing at the equilibrium point if it's not (0,0)
        if not np.allclose(equilibrium_point, [0,0]):
            ax.axhline(equilibrium_point[1], color='grey', linestyle='--', linewidth=0.8)
            ax.axvline(equilibrium_point[0], color='grey', linestyle='--', linewidth=0.8)
        else: # If equilibrium is (0,0), draw standard axes
             ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
             ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

        # Add colorbar - увеличен размер шрифта
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # Important for an independent colorbar
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Time evolution', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Setup title with coefficients and eigenvalues
        full_title = title
        subtitle_parts = []
        if k_coeffs:
            subtitle_parts.append(f"$k_1 = {k_coeffs[0]:.2f}, k_2 = {k_coeffs[1]:.2f}$")
        if eigenvalues:
             lambda1_str = f"{eigenvalues[0].real:.2f}{eigenvalues[0].imag:+.2f}i"
             lambda2_str = f"{eigenvalues[1].real:.2f}{eigenvalues[1].imag:+.2f}i"
             # Use raw string (r"") for LaTeX
             subtitle_parts.append(rf"$\lambda_1 = {lambda1_str}, \lambda_2 = {lambda2_str}$")
        if subtitle_parts:
            full_title += "\n" + "\n".join(subtitle_parts)

        # Увеличим размер шрифта заголовка
        ax.set_title(full_title, fontsize=14)

        # Use raw strings (r"") for LaTeX labels с увеличенным размером шрифта
        ax.set_xlabel(r'$\theta$', fontsize=14)
        ax.set_ylabel(r'$\dot{\theta}$', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Увеличим размер меток осей
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Set axis limits
        if plot_range:
            ax.set_xlim(plot_range[0], plot_range[1])
            ax.set_ylim(plot_range[2], plot_range[3])
        else:
            # Automatic limit detection with a small padding
            pad_x = (max(all_theta) - min(all_theta)) * 0.1 if all_theta else 0.1
            pad_y = (max(all_theta_dot) - min(all_theta_dot)) * 0.1 if all_theta_dot else 0.1
            # Ensure pads are not zero if range is zero
            pad_x = max(pad_x, 0.1)
            pad_y = max(pad_y, 0.1)

            min_x = min(all_theta) - pad_x
            max_x = max(all_theta) + pad_x
            min_y = min(all_theta_dot) - pad_y
            max_y = max(all_theta_dot) + pad_y

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

        # Применяем tight_layout для лучшего размещения элементов на графике
        plt.tight_layout()
        return ax # Return axes object 