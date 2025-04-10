import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

def analyze_region(region_type, phase_module_path='phase.py', show_plot=True):
    """
    Analyzes a specific region of a pendulum with linear control.
    Shows the phase portrait and prints additional information as plain text:
    transition matrix, eigenvalues, determinant, trace.
    
    Parameters:
        region_type (int): Region number (1-7)
        phase_module_path (str): Path to the phase.py module
        show_plot (bool): Whether to show the plot or just print information
    
    Returns:
        plt object: Matplotlib plot object
    """
    # Dynamically import the phase.py module
    spec = importlib.util.spec_from_file_location("phase", phase_module_path)
    phase = importlib.util.module_from_spec(spec)
    sys.modules["phase"] = phase
    spec.loader.exec_module(phase)
    
    # Get functions from the module
    get_k_values = phase.get_k_values
    simulate_system = phase.simulate_system
    plot_trajectories = phase.plot_trajectories
    
    # Get k1 and k2 values for the specified region
    k1, k2 = get_k_values(region_type)
    
    # Calculate transition matrix according to theoretical formulation
    A = np.array([
        [0, 1],
        [-1 + k1, k2]
    ])
    
    # Calculate eigenvalues, determinant and trace
    eigenvalues = np.linalg.eigvals(A)
    determinant = np.linalg.det(A)
    trace = np.trace(A)
    
    # Manual determinant calculation for 2x2 matrix
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    manual_det = a*d - b*c
    
    # Region names in English
    region_names = {
        1: "Stable focus", 
        2: "Stable node",
        3: "Critical damping",
        4: "Center (neutral stability)",
        5: "Unstable focus",
        6: "Saddle point",
        7: "Saddle-node"
    }
    
    # Print all the information as plain text
    print(f"===== ANALYSIS OF REGION {region_type} =====")
    print(f"Region type: {region_names.get(region_type, 'Unknown')}")
    print(f"Parameters: k1 = {k1:.4f}, k2 = {k2:.4f}")
    print("\nTransition matrix A:")
    print(f"[[{A[0,0]:.4f}, {A[0,1]:.4f}],")
    print(f" [{A[1,0]:.4f}, {A[1,1]:.4f}]]")
    
    print("\nEigenvalues:")
    for i, eig in enumerate(eigenvalues):
        if abs(eig.imag) < 1e-10:  # Real eigenvalue
            print(f"λ{i+1} = {eig.real:.4f}")
        else:  # Complex eigenvalue
            print(f"λ{i+1} = {eig.real:.4f} {'+' if eig.imag >= 0 else '-'} {abs(eig.imag):.4f}i")
    
    print()
    print(f"Determinant: {determinant:.4f}")
    print()
    
    print(f"Trace: {trace:.4f}")
    print()
    
    # Calculate and print discriminant
    discriminant = trace**2 - 4*determinant
    print(f"Discriminant: {discriminant:.4f}")
    print(f"Discriminant calculation: trace²-4·det = ({trace:.4f})² - 4·({determinant:.4f}) = {trace**2:.4f} - {4*determinant:.4f} = {discriminant:.4f}")
    print()
    # Additional system characteristics
    if determinant > 0 and trace < 0:
        stability = "Asymptotically stable"
    elif determinant > 0 and trace == 0:
        stability = "Neutrally stable (conservative system)"
    elif determinant > 0 and trace > 0:
        stability = "Unstable"
    elif determinant < 0:
        stability = "Unstable (saddle point)"
    else:
        stability = "Requires nonlinear analysis"
    
    print(f"Stability: {stability}")
    
    # Print eigenvalue characteristics
    if determinant > 0:
        if discriminant < 0:
            eigen_type = "Complex conjugate (oscillatory behavior)"
        elif discriminant > 0:
            eigen_type = "Real and distinct (non-oscillatory behavior)"
        else:
            eigen_type = "Real and repeated (critical damping)"
        print(f"Eigenvalue type: {eigen_type}")
    
    # Only show plot if requested
    if show_plot:
        # Disable LaTeX rendering to avoid issues
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif"
        })
        
        # Simulate the system
        simulation_result = simulate_system(region_type)
        
        # Plot trajectories
        plt_obj = plot_trajectories(simulation_result)
        
        # Add a title with region information without LaTeX
        region_title = f"Region {region_type}: {region_names.get(region_type, 'Unknown')} (k1={k1:.2f}, k2={k2:.2f})"
        plt_obj.suptitle(region_title, fontsize=12)
        
        return plt_obj
    
    return None

# Example usage:
# analyze_region(1)  # Analysis of region 1 (stable focus)