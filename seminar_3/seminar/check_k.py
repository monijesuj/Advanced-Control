import importlib.util
import sys
from phase import *
import numpy as np



def check_k(region_type):
    # Dynamically import the phase.py module

    
    # Get functions from the module
    
    # Get k1 and k2 values for the specified region
    k1, k2 = get_k_values(region_type)
    
    # Calculate transition matrix according to theoretical formulation
    A = np.array([
        [0, 1],
        [-1 + k1, k2]
    ])

    eigenvalues = np.linalg.eigvals(A)
    determinant = np.linalg.det(A)
    trace = np.trace(A)

    region_names = {
        1: "Stable focus", 
        2: "Stable node",
        3: "Critical damping",
        4: "Center (neutral stability)",
        5: "Unstable focus",
        6: "Saddle point",
        7: "Saddle-node"
    }

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
    
    print(f"\nDeterminant: {determinant:.4f}")
    print(f"Trace: {trace:.4f}")
    
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
        if trace**2 - 4*determinant < 0:
            eigen_type = "Complex conjugate (oscillatory behavior)"
        elif trace**2 - 4*determinant > 0:
            eigen_type = "Real and distinct (non-oscillatory behavior)"
        else:
            eigen_type = "Real and repeated (critical damping)"
        print(f"Eigenvalue type: {eigen_type}")
