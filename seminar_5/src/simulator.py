import numpy as np
from tqdm import tqdm # Optional: for progress bar
import typing # Import typing for Type hint

from .system import System
from .controller import Controller
from .plotter import Plotter

class Simulator:
    def __init__(self, system: System, controller: Controller, dt: float, num_steps: int):
        """
        Initializes the Simulation.

        Args:
            system: The system to simulate.
            controller: The controller to use.
            dt: Simulation time step.
            num_steps: Total number of simulation steps.
        """
        self.system = system
        self.controller = controller
        self.dt = dt
        self.num_steps = num_steps
        self.time_vector = np.linspace(0, dt * num_steps, num_steps + 1)

        # History storage
        # Get state dimension dynamically
        initial_state = self.system.get_state()
        state_dim = initial_state.shape[0]
        self.state_history = np.zeros((num_steps + 1, state_dim))
        # Control history stores [control_value, controller_index] or [control_value, nan]
        # Initialize with nan to indicate missing index by default
        self.control_history = np.full((num_steps, 2), np.nan) 
        self.time_history = np.zeros(num_steps + 1)

    def run(self):
        """Runs the simulation loop."""
        # Store initial state
        self.state_history[0, :] = self.system.get_state()
        self.time_history[0] = 0

        # Use tqdm range description if tqdm is available
        step_range = range(self.num_steps)
        if 'tqdm' in globals():
            step_range = tqdm(step_range, desc="Simulation Progress", leave=False) # leave=False for nested loops

        for i in step_range:
            current_time = self.time_vector[i]
            # 1. Compute control input 
            # Controller can return scalar (control_value) or 2-element array/list [control_value, index]
            control_output = self.controller.compute_control(self.system, current_time)
            
            # Ensure control_output is a numpy array for consistent handling
            control_output = np.asarray(control_output)

            if control_output.ndim == 0: # Scalar control value
                control_input = control_output.item() # Extract scalar value
                self.control_history[i, 0] = control_input
                # Index remains np.nan as initialized
            elif control_output.ndim == 1 and control_output.shape == (2,): # [control_value, index]
                control_input = control_output[0]
                self.control_history[i, :] = control_output # Store both value and index
            else:
                 raise ValueError(f"Controller must return a scalar or a 2-element vector [control, index]. Got: {control_output}")

            # 2. Apply control and step the system
            self.system.step(self.dt, control_input)

            # 3. Store results
            self.state_history[i + 1, :] = self.system.get_state()
            self.time_history[i + 1] = current_time + self.dt

    def plot_results(self):
        """Plots the simulation results."""
        # Pass the entire control history (value and index) to the Plotter.
        # Note: Plotter class must be updated to handle the 2D control_history array.
        plotter = Plotter(
            self.time_history, 
            self.state_history, 
            self.control_history, # Pass the combined [control_value, controller_index] history
            self.system
            # Removed separate index argument
        )
        plotter.plot_results()

    def get_results(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the simulation results (time, state, control_vector)."""
        return self.time_history, self.state_history, self.control_history

    @classmethod
    def run_multiple(
        cls,
        initial_states_list: list[np.ndarray],
        system_class: typing.Type[System],
        system_args: dict,
        controller_class: typing.Type[Controller],
        controller_args: dict,
        dt: float,
        num_steps: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Runs multiple simulations for a list of initial states.

        Args:
            cls: The class itself (automatically passed by @classmethod).
            initial_states_list: List of initial state NumPy arrays.
                                 For LightHouseKeeper, expects shape (2,) [pos, vel].
            system_class: The class of the system to simulate (e.g., Pendulum).
            system_args: Dictionary of arguments to pass to the system constructor
                         (excluding initial state related args like 'initial_state', 'initial_position', 'initial_velocity').
            controller_class: The class of the controller to use.
            controller_args: Dictionary of arguments to pass to the controller constructor.
            dt: Simulation time step.
            num_steps: Total number of simulation steps for each trajectory.

        Returns:
            A list of tuples, where each tuple contains (time_history, state_history)
            for one simulation run.
        """
        results_list = []
        print(f"Running {len(initial_states_list)} simulations...")

        # Use tqdm for the outer loop if available
        outer_loop_range = range(len(initial_states_list))
        if 'tqdm' in globals():
             outer_loop_range = tqdm(outer_loop_range, desc="Overall Progress")

        for i in outer_loop_range:
            initial_state = initial_states_list[i]
            if not isinstance(initial_state, np.ndarray) or initial_state.ndim != 1:
                raise ValueError(f"Element {i} in initial_states_list is not a 1D NumPy array.")

            # Prepare arguments for system constructor
            current_system_args = system_args.copy()

            # --- Handle system-specific initial state arguments --- 
            # Check if the system is LightHouseKeeper based on its name to avoid import issues
            if system_class.__name__ == 'LightHouseKeeper':
                if initial_state.shape != (2,):
                     raise ValueError(f"Initial state for LightHouseKeeper must have shape (2,) [pos, vel]. Got {initial_state.shape}")
                current_system_args['initial_position'] = initial_state[0]
                current_system_args['initial_velocity'] = initial_state[1]
            else: # Assume generic system accepts 'initial_state' argument
                current_system_args['initial_state'] = initial_state
            # --------------------------------------------------------

            # Create system instance with the specific initial state arguments
            system = system_class(**current_system_args)

            # Create controller instance
            controller = controller_class(**controller_args)

            # Create simulation instance using cls
            simulation = cls(
                system=system,
                controller=controller,
                dt=dt,
                num_steps=num_steps
            )

            # Run the individual simulation (no progress bar here)
            simulation.run() # Consider adding a quiet mode to sim.run()
            # Get results, control_history now contains vectors
            time_hist, state_hist, _ = simulation.get_results()

            results_list.append((time_hist, state_hist))

        print(f"{len(initial_states_list)} simulations finished.")
        return results_list 