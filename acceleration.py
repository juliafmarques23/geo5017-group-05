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

        #update a, v and p0
        a = a - (learning_rate * grad_a)
        v = v - (learning_rate * grad_v)
        p0 = p0 - (learning_rate * grad_p0)

    final_error = np.sum(((0.5 * a * t**2 + v * t + p0) - positions)**2)
    print(f"Results for {label}")
    print(f"Velocity (v_{label.lower()}): {v:.4f}")
    print(f"Initial Pos (p0_{label.lower()}): {p0:.4f}")
    print(f"Residual Error: {final_error:.4f}\n")
    return a, v, p0, final_error

# input data
t = np.array([1, 2, 3, 4, 5, 6])
x_data = np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57])
y_data = np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91])
z_data = np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])


ax, vx, p0x, err_x = constant_acceleration(t, x_data, "X")
ay, vy, p0y, err_y = constant_acceleration(t, y_data, "Y")
az, vz, p0z, err_z = constant_acceleration(t, z_data, "Z")

total_residual_error = err_x + err_y + err_z
print(f"Total sum-of-squares error: {total_residual_error:.4f}")

#The total error is lower because we increase the polynomial degree
# of the model from 1 (linear) to 2 (quadratic). we give the acceleration model
# 3 parameters (a, v, p0).
# we could improve the model by collecting more data, instead of 6 points.

#for t=7:
t_next = 7
x7 = 0.5 * ax * t_next**2 + vx * t_next + p0x
y7 = 0.5 * ay * t_next**2 + vy * t_next + p0y
z7 = 0.5 * az * t_next**2 + vz * t_next + p0z
print(f"Predicted Position at t=7: ({x7:.4f}, {y7:.4f}, {z7:.4f})")
