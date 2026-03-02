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


if __name__ == "__main__":

    # Time stamps (1 Hz sampling)
    t = np.array([1, 2, 3, 4, 5, 6])

    # Observed 3D positions
    data = {
        'X': np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57]),
        'Y': np.array([0, 1.68, 1.82, 0.28, -1.51, -1.91]),
        'Z': np.array([1, 2.38, 2.49, 2.15, 2.59, 4.32])
    }

    # Store estimated parameters per axis
    results = {}

    # Fit constant velocity model independently for X, Y, and Z
    for axis, values in data.items():
        vx, p0x, err = constant_velocity(t, values, axis)
        results[axis] = {'v': vx, 'p0': p0x, 'err': err}

    # Total residual error across all dimensions
    total_err = sum(r['err'] for r in results.values())

    # Output results
    print(f"Total Residual Error (Linear): {total_err:.4f}")
    print(f"Velocity Vector: [{results['X']['v']:.4f}, {results['Y']['v']:.4f}, {results['Z']['v']:.4f}]")