import numpy as np
import plotly.graph_objects as go


# --- 1. SOLVER LOGIC ---
def constant_acceleration(t, positions, label, learning_rate=0.001, max_iter=10000):
    """Solves for constant acceleration (a, v, p0) using Gradient Descent."""
    a, v, p0 = 0.0, 0.0, 0.0
    n = len(t)

    for _ in range(max_iter):
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


def plot_comparison(measured, predicted):
    """Visualizes measured telemetry against predicted trajectory."""
    mx, my, mz = zip(*measured)
    px, py, pz = zip(*predicted)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=mx, y=my, z=mz, mode='markers',
                               marker=dict(size=5, color='red'), name='Measured Data'))
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers',
                               marker=dict(size=3, color='blue'), name='Model Prediction'))
    fig.show()


if __name__ == "__main__":
    # Data definitions
    t = np.array([1, 2, 3, 4, 5, 6])
    x_data = np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57])
    y_data = np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91])
    z_data = np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])

    # Gradient Descent execution
    ax, vx, p0x, err_x = constant_acceleration(t, x_data, "X")
    ay, vy, p0y, err_y = constant_acceleration(t, y_data, "Y")
    az, vz, p0z, err_z = constant_acceleration(t, z_data, "Z")

    print(f"Total Residual Error: {err_x + err_y + err_z:.4f}")

    # Future prediction (t=7)
    t_next = 7


    def predict(t, a, v, p0): return 0.5 * a * t ** 2 + v * t + p0


    x7, y7, z7 = predict(t_next, ax, vx, p0x), predict(t_next, ay, vy, p0y), predict(t_next, az, vz, p0z)
    print(f"Prediction at t=7: ({x7:.4f}, {y7:.4f}, {z7:.4f})")

    # Data preparation for plotting
    measured_data = list(zip(x_data, y_data, z_data))
    predicted_positions = [(predict(i, ax, vx, p0x), predict(i, ay, vy, p0y), predict(i, az, vz, p0z)) for i in t]
    predicted_positions.append((x7, y7, z7))

    plot_comparison(measured_data, predicted_positions)