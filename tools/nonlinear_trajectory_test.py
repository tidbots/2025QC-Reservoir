#!/usr/bin/env python3
"""
Non-linear Trajectory Test: V1/V2/V3/Kalman Comparison
Generate complex non-linear trajectories (periodic, chaotic patterns)
where ESN might outperform Kalman filter.
"""
import sys
import os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.10/site-packages'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, RLS
from datetime import datetime
import argparse


class OnlineStandardizer:
    def __init__(self, mean, var, alpha=0.02):
        self.mean = np.array(mean, dtype=float)
        self.var = np.array(var, dtype=float)
        self.alpha = alpha
        self.eps = 1e-6

    def update(self, x):
        x = np.atleast_2d(x)
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x.mean(axis=0)
        self.var = (1 - self.alpha) * self.var + self.alpha * x.var(axis=0)

    def transform(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.eps)

    def inverse_transform(self, xs):
        return xs * np.sqrt(self.var + self.eps) + self.mean


class SimpleKalmanFilter:
    def __init__(self, dt=0.1, process_noise=0.1, measurement_noise=0.05):
        self.dt = dt
        self.x = np.zeros(4)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4)
        self.initialized = False

    def update(self, z):
        z = np.array(z).flatten()
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            return self.x[:2]
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        return self.x[:2]

    def predict_future(self, steps):
        predictions = []
        x_temp = self.x.copy()
        for _ in range(steps):
            x_temp = self.F @ x_temp
            predictions.append(x_temp[:2].copy())
        return np.array(predictions)


class TrajectoryComplexityAnalyzer:
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.velocity_history = deque(maxlen=history_size)
        self.direction_history = deque(maxlen=history_size)

    def update(self, pos, prev_pos):
        if prev_pos is not None:
            velocity = pos - prev_pos
            self.velocity_history.append(velocity)
            if len(self.velocity_history) >= 2:
                v1 = self.velocity_history[-2]
                v2 = self.velocity_history[-1]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0.001 and norm2 > 0.001:
                    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
                    angle = np.arccos(cos_angle)
                    self.direction_history.append(angle)

    def get_complexity_score(self):
        if len(self.direction_history) < 3:
            return 0.3
        angles = list(self.direction_history)
        avg_angle = np.mean(angles)
        max_angle = np.max(angles)
        speeds = [np.linalg.norm(v) for v in self.velocity_history]
        if len(speeds) > 1 and np.mean(speeds) > 0.001:
            speed_cv = np.std(speeds) / np.mean(speeds)
        else:
            speed_cv = 0
        direction_score = min(1.0, (avg_angle + max_angle * 0.5) / (np.pi * 0.5))
        speed_score = min(1.0, speed_cv * 2)
        return 0.6 * direction_score + 0.4 * speed_score


class AdaptiveWeightController:
    def __init__(self, initial_esn_weight=0.5, learning_rate=0.1):
        self.esn_weight = initial_esn_weight
        self.learning_rate = learning_rate
        self.esn_errors = deque(maxlen=20)
        self.kalman_errors = deque(maxlen=20)

    def update_errors(self, esn_error, kalman_error):
        self.esn_errors.append(esn_error)
        self.kalman_errors.append(kalman_error)

    def get_adaptive_weight(self, complexity_score):
        base_esn_weight = 0.3 + 0.5 * complexity_score
        if len(self.esn_errors) >= 5 and len(self.kalman_errors) >= 5:
            avg_esn_err = np.mean(self.esn_errors)
            avg_kalman_err = np.mean(self.kalman_errors)
            if avg_esn_err + avg_kalman_err > 1e-6:
                perf_ratio = avg_kalman_err / (avg_esn_err + avg_kalman_err)
                adaptive_weight = 0.7 * base_esn_weight + 0.3 * perf_ratio
            else:
                adaptive_weight = base_esn_weight
        else:
            adaptive_weight = base_esn_weight
        return np.clip(adaptive_weight, 0.2, 0.9)


def create_diverse_esns(n_models=5, base_units=50, seed=42, rls_forgetting=0.99):
    """Create diverse ESN ensemble with optimized parameters."""
    esns = []
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        units = base_units + int(rng.integers(-5, 6))
        sr = float(rng.uniform(0.90, 0.98))  # Optimized: higher spectral radius
        lr = float(rng.uniform(0.50, 0.70))  # Optimized: higher leaking rate
        input_scaling = float(rng.uniform(0.40, 0.60))  # Optimized: higher input scaling
        bias = float(rng.uniform(-0.1, 0.1))
        reservoir = Reservoir(units=units, sr=sr, lr=lr, input_scaling=input_scaling,
                             bias=bias, seed=int(seed + i))
        readout = RLS(forgetting=rls_forgetting)
        esn = reservoir >> readout
        esns.append(esn)
    return esns


def generate_nonlinear_trajectory(trajectory_type, n_points=300, noise=0.01):
    """Generate complex non-linear trajectory patterns."""

    if trajectory_type == "circle":
        # Circular motion (robot following circular path)
        t = np.linspace(0, 2 * np.pi, n_points)
        radius = 5
        x = radius * np.cos(t)
        y = radius * np.sin(t)

    elif trajectory_type == "figure8":
        # Figure-8 pattern (Lemniscate)
        t = np.linspace(0, 2 * np.pi, n_points)
        scale = 5
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)

    elif trajectory_type == "lissajous":
        # Lissajous curve (3:2 frequency ratio)
        t = np.linspace(0, 2 * np.pi, n_points)
        A, B = 5, 5
        a, b = 3, 2
        delta = np.pi / 2
        x = A * np.sin(a * t + delta)
        y = B * np.sin(b * t)

    elif trajectory_type == "spiral":
        # Expanding spiral
        t = np.linspace(0, 4 * np.pi, n_points)
        r = 0.5 + 0.5 * t
        x = r * np.cos(t)
        y = r * np.sin(t)

    elif trajectory_type == "sinusoidal":
        # Sinusoidal wave motion
        t = np.linspace(0, 4 * np.pi, n_points)
        x = t
        y = 3 * np.sin(t)

    elif trajectory_type == "double_sin":
        # Double sinusoidal (two frequencies)
        t = np.linspace(0, 4 * np.pi, n_points)
        x = t
        y = 2 * np.sin(t) + 1.5 * np.sin(3 * t)

    elif trajectory_type == "rosette":
        # Rosette pattern (5-petals)
        t = np.linspace(0, 2 * np.pi, n_points)
        k = 5
        r = 5 * np.cos(k * t)
        x = r * np.cos(t)
        y = r * np.sin(t)

    elif trajectory_type == "lorenz_2d":
        # Lorenz attractor projected to 2D
        # Simplified Lorenz-like chaotic behavior
        dt = 0.01
        sigma, rho, beta = 10, 28, 8/3
        x_lorenz = np.zeros(n_points)
        y_lorenz = np.zeros(n_points)
        z_lorenz = np.zeros(n_points)
        x_lorenz[0], y_lorenz[0], z_lorenz[0] = 1, 1, 1
        for i in range(1, n_points):
            dx = sigma * (y_lorenz[i-1] - x_lorenz[i-1])
            dy = x_lorenz[i-1] * (rho - z_lorenz[i-1]) - y_lorenz[i-1]
            dz = x_lorenz[i-1] * y_lorenz[i-1] - beta * z_lorenz[i-1]
            x_lorenz[i] = x_lorenz[i-1] + dx * dt
            y_lorenz[i] = y_lorenz[i-1] + dy * dt
            z_lorenz[i] = z_lorenz[i-1] + dz * dt
        # Project to 2D and scale
        x = x_lorenz / 10
        y = y_lorenz / 10

    elif trajectory_type == "pendulum":
        # Non-linear pendulum motion (not simple harmonic)
        dt = 0.05
        g, L = 9.8, 1.0
        theta = np.zeros(n_points)
        omega = np.zeros(n_points)
        theta[0] = np.pi * 0.8  # Large initial angle
        omega[0] = 0
        for i in range(1, n_points):
            omega[i] = omega[i-1] - (g/L) * np.sin(theta[i-1]) * dt
            theta[i] = theta[i-1] + omega[i] * dt
        x = L * np.sin(theta) * 5
        y = -L * np.cos(theta) * 5 + 5

    elif trajectory_type == "accelerate_decelerate":
        # Acceleration then deceleration (robot motion profile)
        t = np.linspace(0, 1, n_points)
        # S-curve velocity profile
        v = np.where(t < 0.5, 2 * t**2, 1 - 2 * (1 - t)**2)
        x = np.cumsum(v) * 0.1
        y = np.sin(t * 4 * np.pi) * 2

    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    # Add noise
    x += np.random.randn(n_points) * noise
    y += np.random.randn(n_points) * noise

    return np.column_stack([x, y])


def evaluate_all_methods(traj, warmup=10, window=20, n_models=10, future_horizon=20, seed=42):
    """Evaluate V1, V2, V3, and Kalman on a trajectory."""

    if len(traj) < warmup + future_horizon + 10:
        return None

    # Initialize
    esns = create_diverse_esns(n_models=n_models, base_units=50, seed=seed)
    kalman = SimpleKalmanFilter(dt=0.1)
    complexity_analyzer = TrajectoryComplexityAnalyzer(history_size=10)
    weight_controller = AdaptiveWeightController(initial_esn_weight=0.5)

    warmup_buffer = traj[:warmup]
    X_warm = warmup_buffer[:-1]
    Y_warm = warmup_buffer[1:] - warmup_buffer[:-1]

    scaler_in = StandardScaler().fit(X_warm)
    scaler_tg = StandardScaler().fit(Y_warm)

    in_online_std = OnlineStandardizer(scaler_in.mean_, scaler_in.var_, alpha=0.02)
    tg_online_std = OnlineStandardizer(scaler_tg.mean_, scaler_tg.var_, alpha=0.02)

    X_w_s = in_online_std.transform(X_warm)
    Y_w_s = tg_online_std.transform(Y_warm)

    for esn in esns:
        esn.partial_fit(X_w_s, Y_w_s)

    for i, pos in enumerate(warmup_buffer):
        kalman.update(pos)
        if i > 0:
            complexity_analyzer.update(pos, warmup_buffer[i-1])

    # Config
    adapt_window = 5
    delta_window = 6
    adapt_damping = 0.35
    err_clip = 1.0

    # Storage
    errors = {'v1': [], 'v2': [], 'v3': [], 'kalman': [], 'linear': []}

    window_buffer = list(warmup_buffer)
    prev_pos = warmup_buffer[-1] if len(warmup_buffer) > 0 else None

    for frame_idx in range(warmup, len(traj) - future_horizon):
        pos = traj[frame_idx]
        window_buffer.append(pos)
        if len(window_buffer) > window:
            window_buffer.pop(0)

        kalman.update(pos)
        complexity_analyzer.update(pos, prev_pos)
        in_online_std.update(pos.reshape(1, -1))

        complexity = complexity_analyzer.get_complexity_score()

        X_recent = np.array(window_buffer[-adapt_window:])
        X_recent_s = in_online_std.transform(X_recent)

        # ESN predictions
        all_esn_preds = []
        for esn in esns:
            last_input = X_recent_s.copy()
            future_preds = []
            for _ in range(future_horizon):
                try:
                    delta_s = esn.run(last_input)[-1]
                except:
                    delta_s = np.zeros((1, 2))
                delta = tg_online_std.inverse_transform(delta_s.reshape(1, -1))[0]
                last_pos_esn = in_online_std.inverse_transform(last_input[-1].reshape(1, -1))[0]
                next_pos = last_pos_esn + delta
                future_preds.append(next_pos)
                last_input = np.roll(last_input, -1, axis=0)
                last_input[-1] = in_online_std.transform(next_pos.reshape(1, -1))[0]
            all_esn_preds.append(future_preds)

            # Adaptation
            if len(X_recent_s) > 1:
                fit_len = min(delta_window, len(X_recent_s) - 1)
                adapt_X = X_recent_s[-fit_len-1:-1]
                adapt_Y = np.diff(X_recent_s[-fit_len-1:], axis=0)
                adapt_Y_adj = np.clip(adapt_Y * adapt_damping, -err_clip, err_clip)
                try:
                    esn.partial_fit(adapt_X, adapt_Y_adj)
                except:
                    pass

        if len(window_buffer) >= 2:
            last_delta = np.array(window_buffer[-1]) - np.array(window_buffer[-2])
            tg_online_std.update(last_delta.reshape(1, -1))

        esn_pred = np.mean(np.array(all_esn_preds), axis=0)
        kalman_pred = kalman.predict_future(future_horizon)

        # Linear extrapolation
        if len(window_buffer) >= 2:
            velocity = np.array(window_buffer[-1]) - np.array(window_buffer[-2])
            linear_pred = pos + velocity * future_horizon
        else:
            linear_pred = pos

        # Ground truth
        gt_pos = traj[frame_idx + future_horizon]

        # V1: ESN only
        v1_final = esn_pred[-1]
        v1_error = np.linalg.norm(v1_final - gt_pos)

        # V2: Fixed hybrid (0.7 ESN, 0.3 Kalman)
        v2_final = 0.7 * esn_pred[-1] + 0.3 * kalman_pred[-1]
        v2_error = np.linalg.norm(v2_final - gt_pos)

        # V3: Adaptive
        esn_weight = weight_controller.get_adaptive_weight(complexity)
        v3_final = esn_weight * esn_pred[-1] + (1 - esn_weight) * kalman_pred[-1]
        v3_error = np.linalg.norm(v3_final - gt_pos)

        # Kalman only
        kalman_final = kalman_pred[-1]
        kalman_error = np.linalg.norm(kalman_final - gt_pos)

        # Linear
        linear_error = np.linalg.norm(linear_pred - gt_pos)

        # Update adaptive weight controller
        weight_controller.update_errors(v1_error, kalman_error)

        errors['v1'].append(v1_error)
        errors['v2'].append(v2_error)
        errors['v3'].append(v3_error)
        errors['kalman'].append(kalman_error)
        errors['linear'].append(linear_error)

        prev_pos = pos

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/nonlinear')
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--future_horizon', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    trajectory_types = [
        ("circle", "円周（周期運動）"),
        ("figure8", "8の字"),
        ("lissajous", "リサジュー曲線"),
        ("spiral", "螺旋"),
        ("sinusoidal", "正弦波"),
        ("double_sin", "二重正弦波"),
        ("rosette", "バラ曲線"),
        ("lorenz_2d", "ローレンツ2D"),
        ("pendulum", "非線形振り子"),
        ("accelerate_decelerate", "加減速"),
    ]

    print("=" * 80)
    print("Non-linear Trajectory Test: V1/V2/V3/Kalman/Linear Comparison")
    print(f"Future Horizon: {args.future_horizon} steps")
    print("=" * 80)

    all_results = {}

    for traj_type, traj_name in trajectory_types:
        print(f"\n### {traj_name} ({traj_type}) ###")

        v1_all, v2_all, v3_all, kalman_all, linear_all = [], [], [], [], []

        for trial in range(args.n_trials):
            np.random.seed(trial * 100)
            trajectory = generate_nonlinear_trajectory(traj_type, n_points=300)
            errors = evaluate_all_methods(trajectory, seed=trial, future_horizon=args.future_horizon)

            if errors and len(errors['v1']) > 0:
                v1_all.append(np.mean(errors['v1']))
                v2_all.append(np.mean(errors['v2']))
                v3_all.append(np.mean(errors['v3']))
                kalman_all.append(np.mean(errors['kalman']))
                linear_all.append(np.mean(errors['linear']))
                print(f"  Trial {trial+1}: Kalman={np.mean(errors['kalman']):.3f}, "
                      f"V3={np.mean(errors['v3']):.3f}, V1={np.mean(errors['v1']):.3f}, "
                      f"Linear={np.mean(errors['linear']):.3f}")

        if len(v1_all) > 0:
            all_results[traj_type] = {
                'name': traj_name,
                'v1': np.mean(v1_all),
                'v2': np.mean(v2_all),
                'v3': np.mean(v3_all),
                'kalman': np.mean(kalman_all),
                'linear': np.mean(linear_all)
            }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Non-linear Trajectory Results")
    print("=" * 80)
    header = f"{'Trajectory':<18} {'Kalman':<10} {'V3':<10} {'V2':<10} {'V1':<10} {'Linear':<10} {'Best':<10}"
    print(header)
    print("-" * 80)

    esn_wins = 0
    kalman_wins = 0

    for traj_type, data in all_results.items():
        methods = {'Kalman': data['kalman'], 'V3': data['v3'],
                   'V2': data['v2'], 'V1': data['v1'], 'Linear': data['linear']}
        best = min(methods, key=methods.get)

        if best in ['V1', 'V2', 'V3']:
            esn_wins += 1
        elif best == 'Kalman':
            kalman_wins += 1

        print(f"{data['name']:<18} {data['kalman']:<10.3f} {data['v3']:<10.3f} "
              f"{data['v2']:<10.3f} {data['v1']:<10.3f} {data['linear']:<10.3f} {best:<10}")

    print("-" * 80)
    print(f"ESN (V1/V2/V3) wins: {esn_wins}, Kalman wins: {kalman_wins}")

    # Find trajectories where ESN variants beat Kalman
    print("\n" + "=" * 80)
    print("ESN vs Kalman Detailed Comparison")
    print("=" * 80)
    for traj_type, data in all_results.items():
        kalman_err = data['kalman']
        v3_err = data['v3']
        diff_pct = (kalman_err - v3_err) / kalman_err * 100 if kalman_err > 0 else 0
        status = "V3 BETTER" if v3_err < kalman_err else "Kalman better"
        print(f"{data['name']:<18}: Kalman={kalman_err:.3f}, V3={v3_err:.3f}, diff={diff_pct:+.1f}% [{status}]")

    # Visualization
    n_plots = len(all_results)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, (traj_type, data) in enumerate(all_results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        methods = ['Kalman', 'V3', 'V2', 'V1', 'Linear']
        values = [data['kalman'], data['v3'], data['v2'], data['v1'], data['linear']]
        min_val = min(values)
        colors = ['green' if v == min_val else 'steelblue' for v in values]

        bars = ax.bar(methods, values, color=colors)
        ax.set_title(data['name'], fontsize=10)
        ax.set_ylabel('Mean Error')
        ax.tick_params(axis='x', labelsize=8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Non-linear Trajectory: Method Comparison (Horizon={args.future_horizon})', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f'nonlinear_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")

    # Trajectory visualization
    fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
    axes2 = axes2.flatten()

    for idx, (traj_type, traj_name) in enumerate(trajectory_types):
        if idx >= len(axes2):
            break
        ax = axes2[idx]
        np.random.seed(42)
        traj = generate_nonlinear_trajectory(traj_type, n_points=300)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.7)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8, label='End')
        ax.set_title(traj_name, fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Non-linear Trajectory Patterns', fontsize=14)
    plt.tight_layout()

    traj_path = os.path.join(args.output, f'nonlinear_trajectories_{timestamp}.png')
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory patterns saved: {traj_path}")


if __name__ == "__main__":
    main()
