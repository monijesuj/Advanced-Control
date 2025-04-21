from abc import ABC, abstractmethod
import numpy as np
from .system import System
from .pendulum import Pendulum

class Controller(ABC):
    @abstractmethod
    def compute_control(self, system: System | None = None, t: float | None = None) -> float:
        """Computes the control input."""
        pass

class EnergyControl(Controller):
    def __init__(self, max_torque: float):
        """
        Initializes the Energy-based controller.

        Args:
            max_torque: Maximum allowed torque (tau_bar).
        """
        self.max_torque = max_torque

    def compute_control(self, system: System, t: float | None = None) -> float:
        """
        Computes the control input based on the energy difference.
        Modified to handle theta_dot = 0 at the start.

        Args:
            system: The system instance (must be a Pendulum instance for this controller).
            t: Current time (not used).

        Returns:
            The computed control torque.
        """
        if not isinstance(system, Pendulum):
            raise TypeError("EnergyControl requires a Pendulum system instance.")

        state = system.get_state()
        theta, theta_dot = state

        E_tot = system.get_energy(state)
        E_des = system.get_desired_energy()
        delta_E = E_des - E_tot

        # If delta_E is close to zero, no control needed
        if np.isclose(delta_E, 0):
             return 0.0

        # --- Modified control law --- 
        # If velocity is significant, use the standard law
        # If velocity is near zero, apply torque based on energy error direction
        velocity_threshold = 1e-4
        if abs(theta_dot) > velocity_threshold:
            control_input = self.max_torque * np.sign(delta_E * theta_dot)
        else:
            # Apply torque to increase energy (push away from stable equilibrium)
            # Use sign of delta_E: if delta_E > 0 (need energy), apply positive torque.
            control_input = self.max_torque * np.sign(delta_E)

        return control_input

class LinearFeedbackController(Controller):
    def __init__(self, K1: float, K2: float, target_state: np.ndarray, max_control: float | None = None):
        """
        Initializes the Linear Feedback Controller.

        Args:
            K1: Feedback gain for position error (Torque coefficient).
            K2: Feedback gain for velocity error (Torque coefficient).
            target_state: The desired equilibrium state [theta_target, theta_dot_target].
            max_torque: Optional maximum torque limit. Default: None (no limit).
        """
        self.K1 = K1
        self.K2 = K2
        if target_state.shape != (2,):
            raise ValueError("Target state must be a NumPy array of shape (2,)")
        self.target_state = target_state
        self.max_control = max_control

    def compute_control(self, system: System, t: float | None = None) -> float:
        """
        Computes the control input torque based on linear feedback.
        tau = K1 * (theta - theta_target) + K2 * (theta_dot - theta_dot_target)

        Args:
            system: The system instance.
            t: Current time (not used).

        Returns:
            The computed control torque (tau).
        """
        current_state = system.get_state()
        theta, theta_dot = current_state
        theta_target, theta_dot_target = self.target_state

        error_theta = theta - theta_target
        error_theta_dot = theta_dot - theta_dot_target

        # Wrap angle error around pi? No, linearisation assumes small deviation from target.
        # error_theta = (error_theta + np.pi) % (2 * np.pi) - np.pi # Is this needed? Probably not for stabilization

        control_torque = self.K1 * error_theta + self.K2 * error_theta_dot

        # Apply torque limits if max_torque is specified
        if self.max_control is not None:
            control_torque = np.clip(control_torque, -self.max_control, self.max_control)

        return control_torque

# Ensure K1, K2, target_state are defined before this class if needed globally
# Or pass them during instantiation

class EnergyPDController(Controller):
    def __init__(self,
                 energy_controller: EnergyControl,
                 linear_controller: LinearFeedbackController,
                 eps_theta_switch: float = 0.2,
                 eps_E_switch_factor: float = 0.01):
        """
        Combined controller: Uses a provided Energy controller for swing-up and
        switches to a provided Linear Feedback controller near the target.

        Args:
            energy_controller: An instance of the EnergyControl class.
            linear_controller: An instance of the LinearFeedbackController class.
                               The target state for switching is taken from this controller.
            eps_theta_switch: Angle threshold (rad) relative to linear_controller.target_state[0]
                              for switching to the linear controller. Default: 0.2.
            eps_E_switch_factor: Factor for energy threshold (switch when E_tot > (1 - factor) * E_des).
                               Default: 0.01. E_des is determined from the system.
        """
        if not isinstance(energy_controller, EnergyControl):
            raise TypeError("energy_controller must be an instance of EnergyControl")
        if not isinstance(linear_controller, LinearFeedbackController):
            raise TypeError("linear_controller must be an instance of LinearFeedbackController")
        if not (0 < eps_E_switch_factor < 1):
            raise ValueError("eps_E_switch_factor must be between 0 and 1")
        if eps_theta_switch <= 0:
            raise ValueError("eps_theta_switch must be positive")

        # Store the provided controllers and parameters
        self.energy_controller = energy_controller
        self.linear_controller = linear_controller
        self.eps_theta_switch = eps_theta_switch
        self.eps_E_switch_factor = eps_E_switch_factor

        # Extract necessary info from controllers
        self.target_state = self.linear_controller.target_state # Target for switching check
        self.max_torque = self.energy_controller.max_torque   # For final saturation

        # Internal state
        self.switched_to_linear = False
        self._E_des = None # Will be calculated on first call
        self._eps_E_abs = None # Will be calculated
        self.active_controller_index = 0 # 0: Energy, 1: Linear

    def compute_control(self, system: System, t: float | None = None) -> tuple:
        """
        Computes control input, switching from Energy to Linear control when near target.
        Sets self.active_controller_index to 0 (Energy) or 1 (Linear).
        
        Returns:
            Tuple of (control_torque, active_controller_index): The control torque and the index of the active controller.
        """
        if not isinstance(system, Pendulum):
            raise TypeError("EnergyPDController requires a Pendulum system instance for energy calculations.")

        current_state = system.get_state()
        theta, theta_dot = current_state

        # Calculate E_des and eps_E_abs on first call or if reset
        if self._E_des is None:
            self._E_des = system.get_desired_energy()
            if self._E_des is None or self._E_des <= 0:
                 raise ValueError("Could not get a valid desired energy E_des from the system.")
            self._eps_E_abs = self.eps_E_switch_factor * self._E_des
            # Reset switch state and index when E_des is recalculated
            self.switched_to_linear = False
            self.active_controller_index = 0

        # --- Control Logic ---
        control_torque = 0.0
        if self.switched_to_linear:
            # Already switched, use Linear controller
            control_torque = self.linear_controller.compute_control(system, t)
            self.active_controller_index = 1 # Set index to Linear
        else:
            # Check switching condition
            E_tot = system.get_energy(current_state)
            angle_diff = (theta - self.target_state[0] + np.pi) % (2 * np.pi) - np.pi

            
            angle_condition = abs(angle_diff) < self.eps_theta_switch
            energy_condition = E_tot > (self._E_des - self._eps_E_abs)

            if angle_condition and energy_condition:
                # Switch to Linear controller
                self.switched_to_linear = True
                control_torque = self.linear_controller.compute_control(system, t)
                self.active_controller_index = 1 # Set index to Linear
                # Optionally log the switch event
                # print(f"Switching to Linear control at t={t:.3f}, state=[{theta:.3f}, {theta_dot:.3f}], E_tot={E_tot:.3f}/{self._E_des:.3f}")
            else:
                # Use Energy controller
                control_torque = self.energy_controller.compute_control(system, t)
                self.active_controller_index = 0 # Set index to Energy

        # Apply torque limits (saturation) using max_torque from energy controller
        control_torque = np.clip(control_torque, -self.max_torque, self.max_torque)

        return control_torque, self.active_controller_index

    def reset(self):
        """Resets the switching state for reuse in multiple simulations."""
        self.switched_to_linear = False
        self._E_des = None # Force recalculation of E_des on next call
        self._eps_E_abs = None
        self.active_controller_index = 0 # Reset index
        # Note: This doesn't reset the underlying controllers' internal states if they have any. 