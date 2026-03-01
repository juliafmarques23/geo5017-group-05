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
    data = {
        'X': np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57]),
        'Y': np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91]),
        'Z': np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])
    }

    # Results dictionary
    results = {}
    for axis, values in data.items():
        a, v, p0, err = constant_acceleration(t, values, axis)
        results[axis] = {'a': a, 'v': v, 'p0': p0, 'err': err}

    total_err = sum(r['err'] for r in results.values())
    print(f"Total Residual Error (Acceleration): {total_err:.4f}")

    def get_pos(time, a, v, p0):
        return 0.5 * a * time ** 2 + v * time + p0


    # Create list of [x, y, z] for t=1 through 7
    full_trajectory = [[get_pos(i, results['X']['a'], results['X']['v'], results['X']['p0']),
                        get_pos(i, results['Y']['a'], results['Y']['v'], results['Y']['p0']),
                        get_pos(i, results['Z']['a'], results['Z']['v'], results['Z']['p0'])] for i in range(1, 8)]

    actual_points = list(zip(data['X'], data['Y'], data['Z']))
    new_p = actual_points + [full_trajectory[-1]]

    # Plotting
    trajectory.plot_trajectory(actual_points)  # Actual 1-6
    trajectory.plot_trajectory(new_p)  # Actual 1-6 + Predicted 7
    trajectory.plot_trajectory(full_trajectory)  # Full Model 1-7
