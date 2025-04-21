from abc import ABC, abstractmethod
import numpy as np

class System(ABC):
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Returns the current state vector."""
        pass

    @abstractmethod
    def set_state(self, state: np.ndarray):
        """Sets the current state vector."""
        pass

    @abstractmethod
    def get_state_derivative(self, control_input: float, state: np.ndarray | None = None, t: float | None = None) -> np.ndarray:
        """Computes the derivative of the state vector."""
        pass

    @abstractmethod
    def step(self, dt: float, control_input: float):
        """Performs one simulation step using the Euler method."""
        pass

    @abstractmethod
    def get_kinetic_energy(self, state: np.ndarray | None = None) -> float:
        """
        Computes the kinetic energy of the system.

        Args:
            state: State vector. If None, uses the current state.

        Returns:
            Kinetic energy.
        """
        pass

    @abstractmethod
    def get_potential_energy(self, state: np.ndarray | None = None) -> float:
        """
        Computes the potential energy of the system.

        Args:
            state: State vector. If None, uses the current state.

        Returns:
            Potential energy.
        """
        pass

    @abstractmethod
    def get_energy(self, state: np.ndarray | None = None) -> float:
        """
        Computes the total energy of the system.

        Args:
            state: State vector. If None, uses the current state.

        Returns:
            Total energy.
        """
        pass 