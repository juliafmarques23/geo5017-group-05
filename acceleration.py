import numpy as np
import trajectory


def constant_acceleration(t, positions, label, learning_rate=0.001, max_iter=10000):
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

        # Gradient descent parameter update
        a -= learning_rate * grad_a
        v -= learning_rate * grad_v
        p0 -= learning_rate * grad_p0

    # Final fitting error
    final_error = np.sum(((0.5 * a * t ** 2 + v * t + p0) - positions) ** 2)

    return a, v, p0, final_error


if __name__ == "__main__":

    # Time stamps
    t = np.array([1, 2, 3, 4, 5, 6])

    # Observed 3D positions
    data = {
        'X': np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57]),
        'Y': np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91]),
        'Z': np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])
    }

    # Store fitted parameters per axis
    results = {}

    # Fit motion model independently for X, Y, Z
    for axis, values in data.items():
        a, v, p0, err = constant_acceleration(t, values, axis)
        results[axis] = {'a': a, 'v': v, 'p0': p0, 'err': err}

    # Combined residual error acros all axes
    total_err = sum(r['err'] for r in results.values())
    print(f"Total Residual Error (Acceleration): {total_err:.4f}")

    # Position model function
    def get_pos(time, a, v, p0):
        return 0.5 * a * time ** 2 + v * time + p0

    # Generate full trajectory from t = 1 to 7 (7 is predicted)
    full_trajectory = [[get_pos(i, results['X']['a'], results['X']['v'], results['X']['p0']),
                        get_pos(i, results['Y']['a'], results['Y']['v'], results['Y']['p0']),
                        get_pos(i, results['Z']['a'], results['Z']['v'], results['Z']['p0'])] for i in range(1, 8)]

    # Original observed points (t= 1-6)
    actual_points = list(zip(data['X'], data['Y'], data['Z']))

    # Actual points + predicted point at t = 7
    new_p = actual_points + [full_trajectory[-1]]

    # Plot results
    trajectory.plot_trajectory(actual_points)  # Observed trajectory (1-6)
    trajectory.plot_trajectory(new_p)  # Observed + Predicted (1-7)
    trajectory.plot_trajectory(full_trajectory)  # Full fitted model (1-7)
