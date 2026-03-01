import numpy as np
import trajectory


def constant_acceleration(t, positions, label, learning_rate=0.001, max_iter=10000):
    """Solves for constant acceleration (a, v, p0) using Gradient Descent."""
    a, v, p0 = 0.0, 0.0, 0.0
    n = len(t)

    for it in range(max_iter):
        p_pred = 0.5 * a * t ** 2 + v * t + p0
        error = p_pred - positions

        # Gradient partial derivatives
        grad_a = (2 / n) * np.sum(error * 0.5 * t ** 2)
        grad_v = (2 / n) * np.sum(error * t)
        grad_p0 = (2 / n) * np.sum(error)

        # Update parameters
        a -= learning_rate * grad_a
        v -= learning_rate * grad_v
        p0 -= learning_rate * grad_p0

    final_error = np.sum(((0.5 * a * t ** 2 + v * t + p0) - positions) ** 2)
    return a, v, p0, final_error


if __name__ == "__main__":
    t = np.array([1, 2, 3, 4, 5, 6])
    x_data = np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57])
    y_data = np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91])
    z_data = np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])

    # Unpack 4 values (a, v, p0, error)
    ax, vx, p0x, _ = constant_acceleration(t, x_data, "X")
    ay, vy, p0y, _ = constant_acceleration(t, y_data, "Y")
    az, vz, p0z, _ = constant_acceleration(t, z_data, "Z")


    def get_pos(time, a, v, p0):
        return 0.5 * a * time ** 2 + v * time + p0


    # Create list of [x, y, z] for t=1 through 7
    full_trajectory = []
    for i in range(1, 8):
        full_trajectory.append([get_pos(i, ax, vx, p0x),
                                get_pos(i, ay, vy, p0y),
                                get_pos(i, az, vz, p0z)])

    actual_points = list(zip(x_data, y_data, z_data))
    # Plot using your trajectory.py module
    trajectory.plot_trajectory(actual_points)
    trajectory.plot_trajectory(full_trajectory)








# def plot_trajectory(positions):
#     """Plots 3D line graph with markers on the computed positions"""
#     # Unpack the list of [x, y, z] tuples/lists
#     pos_x, pos_y, pos_z = zip(*positions)
#
#     # Plot graph with points and lines (exactly like the snippet you provided)
#     fig = go.Figure(data=go.Scatter3d(
#         x=pos_x, y=pos_y, z=pos_z,
#         marker=dict(size=3, color='darkblue', symbol='x'),
#         line=dict(color='darkblue', width=2)))
#
#     fig.update_layout(title="Drone Trajectory (t=1 to 7)",
#                       scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
#     fig.show()


# --- 3. MAIN EXECUTION ---
# if __name__ == "__main__":
#     t = np.array([1, 2, 3, 4, 5, 6])
#     x_data = np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57])
#     y_data = np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91])
#     z_data = np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])
#
#     # Solve
#     ax, vx, p0x, err_x = constant_acceleration(t, x_data, "X")
#     ay, vy, p0y, err_y = constant_acceleration(t, y_data, "Y")
#     az, vz, p0z, err_z = constant_acceleration(t, z_data, "Z")
#
#
#     # Predict t=1 to 7
#     def get_pos(time, a, v, p0):
#         return 0.5 * a * time ** 2 + v * time + p0
#
#
#     # Create list of 7 points [x, y, z]
#     full_trajectory = []
#     for i in range(1, 8):
#         full_trajectory.append([get_pos(i, ax, vx, p0x),
#                                 get_pos(i, ay, vy, p0y),
#                                 get_pos(i, az, vz, p0z)])
#
#     print(f"Prediction at t=7: {full_trajectory[-1]}")
#
#     # Plotting using the requested style
#     plot_trajectory(full_trajectory)