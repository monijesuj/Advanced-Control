# Pendulum Adaptive Control

This repository implements a model reference adaptive controller (MRAC) for a simple pendulum, enabling stabilization at any desired angle (e.g., upright) by estimating unknown parameters in real time. The design follows the principles outlined in the provided PDF (`Adaptive_control_basics.pdf`).

---

## Physical Model

- **Mass (m)**: pendulum bob mass (kg)
- **Length (l)**: pendulum rod length (m)
- **Gravity (g)**: 9.81 m/s²

The equation of motion is given by:

![Equation](https://latex.codecogs.com/png.latex?I%20%3D%20m%20l%5E2%2C%20%5Cddot%5Ctheta%20%3D%20-%5Cfrac%7Bg%7D%7Bl%7D%20%5Csin%28%5Ctheta%29%20%2B%20%5Cfrac%7Bu%7D%7Bm%20l%5E2%7D)

Define the tracking errors:

![Errors](https://latex.codecogs.com/png.latex?e%20%3D%20%5Ctheta%20-%20%5Ctheta_%7Bref%7D%2C%20%5Cdot%20e%20%3D%20%5Cdot%5Ctheta%20-%20%5Cdot%5Ctheta_%7Bref%7D)

and the regressor for gravity disturbance:

![Regressor](https://latex.codecogs.com/png.latex?%5Cphi%28%5Ctheta%2C%20%5Ctheta_%7Bref%7D%29%20%3D%20%5Csin%28%5Ctheta%20-%20%5Ctheta_%7Bref%7D%29)

## Adaptive Control Law

We assume an unknown constant parameter \(\theta^*\) associated with the gravity term. The MRAC control law and update are:

![Control Law](https://latex.codecogs.com/png.latex?u%20%3D%20-k_p%20e%20-%20k_d%20%5Cdot%20e%20-%20%5Chat%5Ctheta%20%5Cphi%2C%20%5Cdot%7B%5Chat%5Ctheta%7D%20%3D%20%5Cgamma%20%5Cphi%20e)

Discretized update (with time step \(\Delta t\)):

![Update](https://latex.codecogs.com/png.latex?%5Chat%5Ctheta_%7Bk%2B1%7D%20%3D%20%5Chat%5Ctheta_k%20+%20%5Cgamma%20%5Cphi%28%5Ctheta_k%2C%5Ctheta_%7Bref%7D%29%20e_k%20%5CDelta%20t)

---

## Code Structure

```
├── controllers/
│   └── adaptive_controller.py   # AdaptiveController class
├── simulation/
│   └── simulate.py              # Euler integration, plotting, animation
├── requirements.txt             # numpy, scipy, matplotlib
├── Adaptive_control_basics.pdf   # Reference derivations
└── README.md                    # This file
```

- **controllers/adaptive_controller.py**:
  - `AdaptiveController(m, l, k_p, k_d, gamma)`
  - `set_dt(dt)` to configure Δt for parameter update
  - `control(theta, theta_dot, theta_ref, theta_dot_ref)` returns torque u and updates θ̂

- **simulation/simulate.py**:
  1. Prepends project root to `PYTHONPATH`
  2. Parses CLI arguments:
     - `--theta0` (initial angle, rad)
     - `--theta_ref` (target angle, rad; default π)
  3. Uses dt=0.001 s, T=10 s, runs Euler integration
  4. Records histories of θ, θ̇, u, θ̂
  5. Generates:
     - Time‑series plots: angle vs. time, control torque, parameter estimate
     - 2D animation of pendulum swing

---

## Installation

```bash
pip install -r requirements.txt
```

Supported Python 3.7+.

---

## Usage

```bash
python simulation/simulate.py --theta0 1.2 --theta_ref 3.1416
```

- **--theta0**: initial angle in radians
- **--theta_ref**: desired stabilization angle in radians (upright = π)

Adjust k_p, k_d, γ in `controllers/adaptive_controller.py` to tune performance.

---

## References

- *Adaptive Control Basics*, Samsung Electronics (2025)
- Slotine, Li—Applied Nonlinear Control (for MRAC fundamentals)

---

*End of README*