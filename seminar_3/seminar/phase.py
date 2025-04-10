import numpy as np
import matplotlib.pyplot as plt
import os

def equilibrium_type(k1, k2):
    # Stable focus region (damped oscillations)
    if k1 < 1 and k2 < 0 and k2**2 < 4*(1-k1):
        return 1
    # Stable node region (aperiodic damping)
    elif k1 < 1 and k2 < 0 and k2**2 > 4*(1-k1):
        return 2
    # Critical damping line
    elif k1 < 1 and k2 < 0 and abs(k2**2 - 4*(1-k1)) < 0.01:
        return 3
    # Neutral stability line (center)
    elif k1 < 1 and abs(k2) < 0.01:
        return 4
    # Unstable focus region
    elif k1 < 1 and k2 > 0:
        return 5
    # Saddle point region
    elif k1 > 1:
        return 6
    # Special boundary (saddle-node)
    elif abs(k1 - 1) < 0.01 and k2 <= 0:
        return 7
    else:
        return 0

def get_k_values(region_type):
    """
    Returns example values of k1 and k2 for the specified region type
    """
    if region_type == 1:  # Stable focus
        return 0.5, -0.5
    elif region_type == 2:  # Stable node
        return 0.5, -2.5
    elif region_type == 3:  # Critical damping
        k1 = 0.5
        k2 = -2 * np.sqrt(1 - k1)
        return k1, k2
    elif region_type == 4:  # Neutral stability (center)
        return 0.5, 0.0
    elif region_type == 5:  # Unstable focus
        return 0.5, 0.5
    elif region_type == 6:  # Saddle point
        return 1.5, 0.0
    elif region_type == 7:  # Saddle-node
        return 1.0, -0.5
    else:
        raise ValueError("Unknown region type")

def simulate_system(region_type, num_points=50, num_steps=500, dt=0.05):
    """
    Simulates a dynamic system in matrix form for the selected region.
    
    Parameters:
        region_type (int): Region number (1-7)
        num_points (int): Number of initial points
        num_steps (int): Number of simulation steps
        dt (float): Time step
        
    Returns:
        dict: Dictionary with trajectories and system parameters
    """
    # Get k1 and k2 values for the specified region
    k1, k2 = get_k_values(region_type)
    
    # Create system matrix in the form x' = Ax
    A = np.array([
        [0, 1],
        [-1, k2 * (1 - k1)]
    ])
    
    # Generate 10 initial points around the equilibrium position (0, 0)
    # Use a small radius for initial points
    radius = 0.2
    np.random.seed(42)  # for reproducibility
    
    initial_conditions = []
    for _ in range(num_points):
        # Generate random points in a circle of radius 'radius'
        r = radius * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        initial_conditions.append(np.array([x0, y0]))
    
    # Simulate the system for each initial condition
    trajectories = []
    
    for init_cond in initial_conditions:
        trajectory = [init_cond.copy()]
        x = init_cond.copy()
        
        for _ in range(num_steps):
            # Simple Euler method for integration
            dx = A @ x
            x = x + dx * dt
            trajectory.append(x.copy())
        
        trajectories.append(np.array(trajectory))
    
    # Return simulation results and parameters
    return {
        "region_type": region_type,
        "k1": k1,
        "k2": k2,
        "trajectories": trajectories,
        "dt": dt,
        "num_steps": num_steps
    }

def plot_trajectories(simulation_result, title=None):
    """
    Visualizes trajectories of the dynamic system.
    
    Parameters:
        simulation_result (dict): Result from the simulate_system function
        title (str, optional): Plot title
    """
    # Enable LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    
    plt.figure(figsize=(10, 8))
    
    # Extract trajectories from simulation results
    trajectories = simulation_result["trajectories"]
    region_type = simulation_result["region_type"]
    k1 = simulation_result["k1"]
    k2 = simulation_result["k2"]
    
    # Calculate system matrix and eigenvalues
    A = np.array([
        [0, 1],
        [-1, k2 * (1 - k1)]
    ])
    eigenvalues = np.linalg.eigvals(A)
    
    # Set colors for initial and final points
    start_color = 'blue'
    end_color = 'green'
    
    # Flag to add to legend only once
    start_added_to_legend = False
    end_added_to_legend = False
    
    # Display each trajectory
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], '-', linewidth=1.5)
        
        # Plot start point
        if not start_added_to_legend:
            plt.plot(traj[0, 0], traj[0, 1], 'o', color=start_color, markersize=5, label='Initial points')
            start_added_to_legend = True
        else:
            plt.plot(traj[0, 0], traj[0, 1], 'o', color=start_color, markersize=5)
        
        # Plot end point
        if not end_added_to_legend:
            plt.plot(traj[-1, 0], traj[-1, 1], 's', color=end_color, markersize=5, label='Final points')
            end_added_to_legend = True
        else:
            plt.plot(traj[-1, 0], traj[-1, 1], 's', color=end_color, markersize=5)
    
    # Add equilibrium point with larger size
    plt.plot(0, 0, 'ro', markersize=12, markeredgecolor='black', label='Equilibrium position')
    
    # Configure axes and titles
    if title is None:
        region_names = {
            1: "Stable focus", 
            2: "Stable node",
            3: "Critical damping",
            4: "Center (neutral stability)",
            5: "Unstable focus",
            6: "Saddle point",
            7: "Saddle-node"
        }
        
        # Format eigenvalues for display using LaTeX
        eig_latex = []
        for e in eigenvalues:
            if abs(e.imag) < 1e-10:  # Real eigenvalue
                eig_latex.append(f"${e.real:.3f}$")
            else:  # Complex eigenvalue
                eig_latex.append(f"${e.real:.3f} {'+' if e.imag >= 0 else '-'} {abs(e.imag):.3f}i$")
        
        eig_str = ", ".join(eig_latex)
        title = f"Region {region_type}: {region_names.get(region_type, 'Unknown')} ($k_1={k1:.2f}$, $k_2={k2:.2f}$)\nEigenvalues: [{eig_str}]"
    
    plt.title(title)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Make axes of equal scale
    plt.axis('equal')
    
    return plt

# Example usage:
if __name__ == "__main__":
    # Create directory for images if it doesn't exist
    images_dir = "simulation_images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Simulate the system for each region
    for region in range(1, 8):
        result = simulate_system(region)
        plt = plot_trajectories(result)
        plt.savefig(os.path.join(images_dir, f"region_{region}_simulation.png"), dpi=150)
        plt.close()
        print(f"Simulation for region {region} saved to file {images_dir}/region_{region}_simulation.png")
