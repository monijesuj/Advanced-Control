import numpy as np
from .system import System

class Pendulum(System):
    def __init__(self, 
                 mass: float = 1.0, 
                 length: float = 1.0, 
                 damping: float = 0.0, 
                 gravity: float = 1, 
                 initial_state: np.ndarray = np.array([0.0, 0.0])):
        """
        Initializes the Pendulum system.

        Args:
            mass: Mass of the pendulum (m).
            length: Length of the pendulum (l).
            damping: Damping coefficient (b). Default: 0.0.
            gravity: Acceleration due to gravity (g).
            initial_state: Initial state [theta, theta_dot].
        """
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.state = np.asarray(initial_state, dtype=float)
        if self.state.shape != (2,):
            raise ValueError("Initial state must be a Numpy array of shape (2,)")

    def get_state(self) -> np.ndarray:
        """Returns the current state vector [theta, theta_dot]."""
        return self.state

    def set_state(self, state: np.ndarray):
        """Sets the current state vector."""
        if state.shape != (2,):
            raise ValueError("State must be a Numpy array of shape (2,)")
        self.state = np.asarray(state, dtype=float)

    def get_state_derivative(self, control_input: float, state: np.ndarray | None = None, t: float | None = None) -> np.ndarray:
        """
        Computes the derivative of the pendulum state vector including damping.

        Args:
            control_input: Control input (torque tau).
            state: Current state [theta, theta_dot]. If None, uses the current state.
            t: Current time. Not used in this implementation but kept for compatibility.

        Returns:
            State vector derivative [theta_dot, theta_ddot].
        """
        if state is None:
            state = self.get_state()
        theta, theta_dot = state
        tau = control_input # Control input is the torque tau

        # Dynamics with friction:
        # theta_ddot = -g/l * sin(theta) - (b/(m*l^2)) * theta_dot + tau / (m*l^2)
        d_state_dt = np.zeros_like(state)
        d_state_dt[0] = theta_dot
        d_state_dt[1] = -(self.g / self.l) * np.sin(theta) - (self.b / (self.m * self.l**2)) * theta_dot + tau / (self.m * self.l**2)
        return d_state_dt

    def step(self, dt: float, control_input: float):
        """
        Performs one simulation step using the Euler method.

        Args:
            dt: Time step.
            control_input: Control input (torque tau).
        """
        current_state = self.get_state()
        # Time t is not used, state uses default
        state_derivative = self.get_state_derivative(control_input=control_input) 
        new_state = current_state + state_derivative * dt
        # Normalize angle theta to the range [-pi, pi] for convenience (not strictly necessary for dynamics)
        # new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi
        self.set_state(new_state)

        return new_state

    def get_energy(self, state: np.ndarray | None = None) -> float:
        """
        Computes the total energy of the system (kinetic + potential).

        Args:
            state: State [theta, theta_dot]. If None, uses the current state.

        Returns:
            Total energy E_tot.
        """
        if state is None:
            state = self.get_state()
        return self.get_kinetic_energy(state) + self.get_potential_energy(state)

    def get_kinetic_energy(self, state: np.ndarray | None = None) -> float:
        """Computes the kinetic energy."""
        if state is None:
            state = self.get_state()
        _, theta_dot = state
        E_kin = 0.5 * self.m * (self.l * theta_dot)**2
        return E_kin

    def get_potential_energy(self, state: np.ndarray | None = None) -> float:
        """Computes the potential energy relative to the bottom position."""
        if state is None:
            state = self.get_state()
        theta, _ = state
        E_pot = self.m * self.g * self.l * (1 - np.cos(theta))
        return E_pot

    def get_desired_energy(self) -> float:
        """
        Returns the desired energy (corresponding to the top equilibrium position).
        E_des = m*g*l*(1 - cos(pi)) = 2*m*g*l
        """
        return 2 * self.m * self.g * self.l 