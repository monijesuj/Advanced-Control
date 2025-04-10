import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_stability_regions(save_path=None, show_plot=True, figsize=(10, 8), dpi=300):
    """
    Plot the stability regions of a linear pendulum system with control parameters k1 and k2.
    
    Parameters:
    -----------
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to show the plot.
    figsize : tuple, default=(10, 8)
        Figure size in inches.
    dpi : int, default=300
        Resolution for the saved figure.
        
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define ranges for k1 and k2
    k1_min, k1_max = -0.5, 2.0
    k2_min, k2_max = -3.0, 1.5
    
    # Create a grid of k1 and k2 values
    k1 = np.linspace(k1_min, k1_max, 500)
    k2 = np.linspace(k2_min, k2_max, 500)
    K1, K2 = np.meshgrid(k1, k2)
    
    # Function to determine equilibrium type
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
    
    # Create an array of equilibrium types
    Z = np.zeros_like(K1)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = equilibrium_type(K1[i, j], K2[i, j])
    
    # Define colors for different regions
    colors = [
        'white',           # 0: undefined
        'lightgreen',      # 1: stable focus
        'darkgreen',       # 2: stable node
        'blue',            # 3: critical damping
        'yellow',          # 4: center (neutral stability)
        'lightcoral',      # 5: unstable focus
        'red',             # 6: saddle
        'purple'           # 7: saddle-node
    ]
    
    # Create color map
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    bounds = np.arange(len(colors) + 1) - 0.5
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, len(colors))
    
    # Draw regions on the plane
    ax.imshow(Z, origin='lower', extent=[k1_min, k1_max, k2_min, k2_max], 
               aspect='auto', cmap=cmap, norm=norm, alpha=0.7)
    
    # Add grid and axes
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(x=1, color='black', linestyle='-', alpha=0.5)
    
    # Draw the curve k2^2 = 4(1-k1) to separate node and focus
    k1_curve = np.linspace(k1_min, 1, 100)
    k2_curve = -2 * np.sqrt(1 - k1_curve)
    ax.plot(k1_curve, k2_curve, 'b--', linewidth=2)
    
    # Add region labels with formulas
    ax.text(0.2, -1.0, "1: Stable focus\n(damped oscillations)\n$k_1 < 1, k_2 < 0, k_2^2 < 4(1-k_1)$", ha='center')
    ax.text(0.2, -2.5, "2: Stable node\n(aperiodic damping)\n$k_1 < 1, k_2 < 0, k_2^2 > 4(1-k_1)$", ha='center')
    ax.text(0.3, 0.5, "5: Unstable focus\n(growing oscillations)\n$k_1 < 1, k_2 > 0$", ha='center')
    ax.text(1.5, -1.0, "6: Saddle\n(unstable)\n$k_1 > 1$", ha='center')
    ax.text(0.4, 0.05, "4: Center\n(periodic oscillations)\n$k_1 < 1, k_2 = 0$", ha='center')
    ax.text(0.9, -1.7, "7: Saddle-node\nbifurcation\n$k_1 = 1, k_2 \\leq 0$", ha='center', rotation=90)
    
    # Add axis labels and title
    ax.set_xlabel('$k_1$ (coefficient of $\\theta$)')
    ax.set_ylabel('$k_2$ (coefficient of $\\dot{\\theta}$)')
    ax.set_title('Stability regions of pendulum with linear control $u(t) = k_1\\theta + k_2\\dot{\\theta}$')
    
    # Create legend with formulas
    legend_elements = [
        Line2D([0], [0], color='lightgreen', lw=10, label='1: Stable focus: $k_1 < 1, k_2 < 0, k_2^2 < 4(1-k_1)$'),
        Line2D([0], [0], color='darkgreen', lw=10, label='2: Stable node: $k_1 < 1, k_2 < 0, k_2^2 > 4(1-k_1)$'),
        Line2D([0], [0], color='blue', lw=4, label='3: Critical damping: $k_1 < 1, k_2 < 0, k_2^2 = 4(1-k_1)$'),
        Line2D([0], [0], color='yellow', lw=10, label='4: Center: $k_1 < 1, k_2 = 0$'),
        Line2D([0], [0], color='lightcoral', lw=10, label='5: Unstable focus: $k_1 < 1, k_2 > 0$'),
        Line2D([0], [0], color='red', lw=10, label='6: Saddle: $k_1 > 1$'),
        Line2D([0], [0], color='purple', lw=10, label='7: Saddle-node: $k_1 = 1, k_2 \\leq 0$')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # Add annotations for key lines and points
    ax.annotate('$k_1 = 1$', xy=(1.05, -2.5), xytext=(1.2, -2.5), 
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    ax.annotate('$k_2 = 0$', xy=(-0.3, 0.05), xytext=(-0.3, 0.3), 
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    ax.annotate('$k_2^2 = 4(1-k_1)$', xy=(0.5, -1.5), xytext=(0.4, -1.7), 
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    plt.tight_layout()
    
    # Save figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show the plot if requested
    if show_plot:
        plt.show()
        
    return fig, ax


# Example usage
if __name__ == "__main__":
    # This code only runs if this file is executed directly
    plot_stability_regions(save_path="plane_k1_k2.png")
    print("Stability regions plot created and saved to 'plane_k1_k2.png'")