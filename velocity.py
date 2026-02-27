import numpy as np
import matplotlib.pyplot as plt

def constant_velocity(t, positions, label, learning_rate=0.001, max_iter=10000):
    """
    Solves for constant velocity using Gradient Descent.
    Model: p(t) = v*t +p0
    """
    v = 0.0
    p0 = 0.0
    n  = len(t)
    for it in range(max_iter):
        #y = a * x + b
        p_pred = v * t + p0
        error = p_pred - positions

        # Gradients calculation (partial derivatives)
        grad_v = (2/n) * np.sum(error * t)
        grad_p0 = (2/n) * np.sum(error)

        # Parameter update: learning rate
        v = v - (learning_rate * grad_v)
        p0 = p0 - (learning_rate * grad_p0)

    # Sum of Squared Errors (SSE)
    final_error = np.sum(((v * t + p0) - positions)**2)

    # Output results
    print(f"Results for {label}")
    print(f"Velocity (v_{label.lower()}): {v:.4f}")
    print(f"Initial Pos (p0_{label.lower()}): {p0:.4f}")
    print(f"Residual Error: {final_error:.4f}\n")

    return v, final_error

# Drone tracking data
t = np.array([1, 2, 3, 4, 5, 6])
x_data = np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57])
y_data = np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91])
z_data = np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])

# Execute Gradient Descent for each axis
vx, err_x = constant_velocity(t, x_data, "X")
vy, err_y = constant_velocity(t, y_data, "Y")
vz, err_z = constant_velocity(t, z_data, "Z")

# Combine errors for total residual assessment
total_residual_error = err_x + err_y + err_z

print(f"Final velocity vector: [{vx:.4f}, {vy:.4f}, {vz:.4f}]")
print(f"Total sum-of-squares error: {total_residual_error:.4f}")
