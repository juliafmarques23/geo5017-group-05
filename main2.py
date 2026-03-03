import numpy as np
import trajectory
import velocity
import acceleration

# Position model function
def get_pos(time, a, v, p0):
    return 0.5 * a * time ** 2 + v * time + p0

def main():

    # SAMPLE DATA:

    points = np.array([[2, 0, 1], [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49], [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59], [0.57, -1.91, 4.32]])

    points_dimensions = np.transpose(points)

    time = np.array([1, 2, 3, 4, 5, 6])

    data = {
        'X': points_dimensions[0],
        'Y': points_dimensions[1],
        'Z': points_dimensions[2]}

    # CONSTANT VELOCITY MODEL:

    # Store estimated parameters per axis
    results_constantv = {}

    # Fit constant velocity model independently for X, Y, and Z
    for axis, values in data.items():
        vx, p0x, err = velocity.constant_velocity(time, values, axis)
        results_constantv[axis] = {'v': vx, 'p0': p0x, 'err': err}

    # Total residual error across all dimensions
    total_err_cv = sum(r['err'] for r in results_constantv.values())

    # Output results
    print(f"Total Residual Error (Linear): {total_err_cv:.4f}")
    print(f"Velocity Vector: [{results_constantv['X']['v']:.4f}, {results_constantv['Y']['v']:.4f}, {results_constantv['Z']['v']:.4f}]")

    # CONSTANT ACCELERATION MODEL:

    results_acceleration = {}

    # Fit motion model independently for X, Y, Z
    for axis, values in data.items():
        a, v, p0, err = acceleration.constant_acceleration(time, values, axis)
        results_acceleration[axis] = {'a': a, 'v': v, 'p0': p0, 'err': err}

    # Combined residual error acros all axes
    total_err_a = sum(r['err'] for r in results_acceleration.values())
    print(f"Total Residual Error (Acceleration): {total_err_a:.4f}")

    # PLOT TRAJECTORY + ESTIMATED P7

    # Generate full trajectory from t = 1 to 7 (7 is predicted)
    full_trajectory = [[get_pos(i, results_acceleration['X']['a'], results_acceleration['X']['v'], results_acceleration['X']['p0']),
                        get_pos(i, results_acceleration['Y']['a'], results_acceleration['Y']['v'], results_acceleration['Y']['p0']),
                        get_pos(i, results_acceleration['Z']['a'], results_acceleration['Z']['v'], results_acceleration['Z']['p0'])] for i in range(1, 8)]

    # Original observed points (t= 1-6)
    actual_points = list(zip(data['X'], data['Y'], data['Z']))

    # Actual points + predicted point at t = 7
    new_p = actual_points + [full_trajectory[-1]]

    # Plot results
    #trajectory.plot_trajectory(actual_points)  # Observed trajectory (1-6)
    trajectory.plot_trajectory(new_p)  # Observed + Predicted (1-7)
    trajectory.plot_trajectory(full_trajectory)  # Full fitted model (1-7)

if __name__ == "__main__":
    main()