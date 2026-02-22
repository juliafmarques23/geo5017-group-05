import numpy as np
import matplotlib.pyplot as plt

def constant_acceleration(t, positions, label, learning_rate=0.001, max_iter=10000):
    a = 0.0
    v = 0.0
    p0 = 0.0
    n  = len(t)
    for it in range(max_iter):
        #y = b2 * x**2 + b1 * x + b0
        #p(t) = (1/2)*a*t**2 + v0*t + p0
        p_pred = 0.5 * a * t**2 + v * t + p0
        error = p_pred - positions

        #gradients
        grad_a = (2/n) * np.sum(error * 0.5 * t**2)
        grad_v = (2/n) * np.sum(error * t)
        grad_p0 = (2/n) * np.sum(error)

        #update v and p0
        a = a - (learning_rate * grad_a)
        v = v - (learning_rate * grad_v)
        p0 = p0 - (learning_rate * grad_p0)

    final_error = np.sum(((0.5 * a * t**2 + v * t + p0) - positions)**2)
    print(f"Results for {label}")
    print(f"Velocity (v_{label.lower()}): {v:.4f}")
    print(f"Initial Pos (p0_{label.lower()}): {p0:.4f}")
    print(f"Residual Error: {final_error:.4f}\n")
    return v, final_error

# input data
t = np.array([1, 2, 3, 4, 5, 6])
x_data = np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57])
y_data = np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91])
z_data = np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])


vx, err_x = constant_acceleration(t, x_data, "X")
vy, err_y = constant_acceleration(t, y_data, "Y")
vz, err_z = constant_acceleration(t, z_data, "Z")

total_residual_error = err_x + err_y + err_z

print(f"Final velocity error: [{vx:.4f}, {vy:.4f}, {vz:.4f}]")
print(f"Total sum-of-squares error: {total_residual_error:.4f}")
