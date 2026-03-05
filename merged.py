import numpy as np
import plotly.graph_objects as go

def constant_velocity(t, positions, label, learning_rate=0.001, max_iter=10000):
    """
    Fits a constant velocity motion model:
        p(t) = v * t + p0

    Parameters are estimated using Gradient Descent by minimizing
    the sum-of-squares error between predicted and observed positions.
    """
    v, p0 = 0.0, 0.0
    n = len(t)

    for it in range(max_iter):
        # Predicted positions using current parameters
        p_pred = v * t + p0

        # Residual error
        error = p_pred - positions

        # Gradient of the mean squared error
        grad_v = (2 / n) * np.sum(error * t)
        grad_p0 = (2 / n) * np.sum(error)

        # Gradient descent update step
        v -= learning_rate * grad_v
        p0 -= learning_rate * grad_p0

    # Final sum-of-squares residual error
    final_error = np.sum(((v * t + p0) - positions) ** 2)
    return v, p0, final_error

def constant_acceleration(t, positions, label, learning_rate=0.001, max_iter=100000):
    """
    Fits a constant-acceleration motion model:
        p(t) = 0.5 * a * t^2 + v * t + p0

    Uses Gradient Descent to estimate:
        a  -> acceleration
        v  -> velocity
        p0 -> initial position
    """

    # Initialize parameters
    a, v, p0 = 0.0, 0.0, 0.0
    n = len(t)

    for it in range(max_iter):
        # Predicted positions using current parameters
        p_pred = 0.5 * a * t ** 2 + v * t + p0

        # Residual error
        error = p_pred - positions

        # Gradients of Mean Squared Error according to parameters
        grad_a = (2 / n) * np.sum(error * 0.5 * t ** 2)
        grad_v = (2 / n) * np.sum(error * t)
        grad_p0 = (2 / n) * np.sum(error)

        if np.linalg.norm([grad_a, grad_v, grad_p0]) < 1e-6:
            break

        # Gradient descent parameter update
        a -= learning_rate * grad_a
        v -= learning_rate * grad_v
        p0 -= learning_rate * grad_p0

    # Final fitting error
    final_error = np.sum(((0.5 * a * t ** 2 + v * t + p0) - positions) ** 2)

    return a, v, p0, final_error

def plot_trajectory(observed, predicted, t_pred=7, title="3D Trajectory"):
    # Unpack coordinates
    x_obs, y_obs, z_obs = zip(*observed)
    x_pred, y_pred, z_pred = zip(*predicted)

    fig = go.Figure()

    # Observed points
    fig.add_trace(go.Scatter3d(
        x=x_obs, y=y_obs, z=z_obs,
        mode="lines+markers",
        marker=dict(size=5, color='red'),
        line=dict(width=2, color='red',  dash='dash'),
        name="Observed (t=1-6)"
    ))

    # Predicted/fitted trajectory
    fig.add_trace(go.Scatter3d(
        x=x_pred, y=y_pred, z=z_pred,
        mode="lines+markers",
        marker=dict(size=5, color='darkblue'),
        line=dict(width=2, color='darkblue'),
        name="Fitted Constant Acceleration (t=1-7)"
    ))

    # Highlight 7th predicted point
    fig.add_trace(go.Scatter3d(
        x=[x_pred[-1]], y=[y_pred[-1]], z=[z_pred[-1]],
        mode="markers",
        marker=dict(size=7, color='blue'),
        name=f"Predicted t={t_pred}"
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )

    fig.show()

if __name__ == "__main__":

    # Time steps
    t = np.array([1, 2, 3, 4, 5, 6])

    # Observed data
    data = {
        'X': np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57]),
        'Y': np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91]),
        'Z': np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])
    }

    # Constant Velocity Results
    vel_results = {}
    for axis, values in data.items():
        v, p0, err = constant_velocity(t, values, axis)
        vel_results[axis] = {'v': v, 'p0': p0, 'err': err}

    total_vel_error = sum(r['err'] for r in vel_results.values())

    print("Constant Velocity Model")
    print(f"Velocity vector: "
          f"[{vel_results['X']['v']:.4f}, "
          f"{vel_results['Y']['v']:.4f}, "
          f"{vel_results['Z']['v']:.4f}]")
    print(f"Total residual error: {total_vel_error:.4f}")

    # Constant Acceleration Results
    acc_results = {}
    for axis, values in data.items():
        a, v, p0, err = constant_acceleration(t, values, axis)
        acc_results[axis] = {'a': a, 'v': v, 'p0': p0, 'err': err}

    total_acc_error = sum(r['err'] for r in acc_results.values())

    print("\nConstant Acceleration Model")
    print(f"Total residual error: {total_acc_error:.4f}")

    # Trajectories
    actual_points = list(zip(data['X'], data['Y'], data['Z']))

    def get_pos(t, a, v, p0):
        return 0.5 * a * t**2 + v * t + p0

    full_trajectory = [
        (
            get_pos(i, acc_results['X']['a'], acc_results['X']['v'], acc_results['X']['p0']),
            get_pos(i, acc_results['Y']['a'], acc_results['Y']['v'], acc_results['Y']['p0']),
            get_pos(i, acc_results['Z']['a'], acc_results['Z']['v'], acc_results['Z']['p0'])
        )
        for i in range(1, 8)
    ]

    predicted_points = actual_points + [full_trajectory[-1]]

    # Plot
    plot_trajectory(actual_points, full_trajectory)
    # plot_trajectory(actual_points, "Observed Trajectory (t = 1–6)")
    # plot_trajectory(predicted_points, "Observed + Predicted Position (t = 7)")
    # plot_trajectory(full_trajectory, "Fitted Constant Acceleration Trajectory (t = 1–7)")

