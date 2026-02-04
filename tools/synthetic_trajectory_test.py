#!/usr/bin/env python3
"""
Synthetic Trajectory Test: V1/V2/V3/Kalman Comparison
Generate trajectories with intentional sharp direction changes
to test where ESN might outperform Kalman filter.
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


# Import evaluation functions from eth_v3_adaptive
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


def create_diverse_esns(n_models=5, base_units=25, seed=42, rls_forgetting=0.99):
    esns = []
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        units = base_units + int(rng.integers(-5, 6))
        sr = float(rng.uniform(0.8, 0.9))
        lr = float(rng.uniform(0.35, 0.6))
        input_scaling = float(rng.uniform(0.2, 0.4))
        bias = float(rng.uniform(-0.2, 0.2))
        reservoir = Reservoir(units=units, sr=sr, lr=lr, input_scaling=input_scaling,
                             bias=bias, seed=int(seed + i))
        readout = RLS(forgetting=rls_forgetting)
        esn = reservoir >> readout
        esns.append(esn)
    return esns


def generate_synthetic_trajectory(trajectory_type, n_points=200, noise=0.01):
    """Generate synthetic trajectory with specific patterns."""

    if trajectory_type == "linear":
        # Straight line
        t = np.linspace(0, 1, n_points)
        x = t * 10
        y = t * 5

    elif trajectory_type == "sharp_turn":
        # Walk straight, then sharp 90° turn
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        turn_point = n_points // 2
        for i in range(n_points):
            if i < turn_point:
                x[i] = i * 0.05
                y[i] = 0
            else:
                x[i] = turn_point * 0.05
                y[i] = (i - turn_point) * 0.05

    elif trajectory_type == "zigzag":
        # Multiple direction changes
        x = np.linspace(0, 10, n_points)
        y = np.zeros(n_points)
        segment_len = n_points // 6
        y_pos = 0
        direction = 1
        for i in range(n_points):
            y[i] = y_pos
            y_pos += direction * 0.03
            if i > 0 and i % segment_len == 0:
                direction *= -1

    elif trajectory_type == "sudden_stop":
        # Walk, stop suddenly, then reverse
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        phase1 = n_points // 3
        phase2 = 2 * n_points // 3
        for i in range(n_points):
            if i < phase1:
                x[i] = i * 0.05
                y[i] = i * 0.02
            elif i < phase2:
                x[i] = phase1 * 0.05
                y[i] = phase1 * 0.02
            else:
                x[i] = phase1 * 0.05 - (i - phase2) * 0.03
                y[i] = phase1 * 0.02 - (i - phase2) * 0.04

    elif trajectory_type == "curved":
        # Smooth curve
        theta = np.linspace(0, np.pi, n_points)
        x = 5 * np.cos(theta)
        y = 5 * np.sin(theta)

    elif trajectory_type == "random_walk":
        # Random walk with sudden direction changes
        np.random.seed(42)
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        vx, vy = 0.05, 0.02
        for i in range(1, n_points):
            if np.random.random() < 0.15:
                angle = np.random.uniform(-np.pi, np.pi)
                speed = np.sqrt(vx**2 + vy**2)
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)
            x[i] = x[i-1] + vx
            y[i] = y[i-1] + vy
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    # Add noise
    x += np.random.randn(n_points) * noise
    y += np.random.randn(n_points) * noise

    return np.column_stack([x, y])


def evaluate_all_methods(traj, warmup=5, window=20, n_models=10, future_horizon=20, seed=42):
    """Evaluate V1, V2, V3, and Kalman on a trajectory."""

    if len(traj) < warmup + future_horizon + 10:
        return None

    # Initialize
    esns = create_diverse_esns(n_models=n_models, base_units=25, seed=seed)
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
    errors = {'v1': [], 'v2': [], 'v3': [], 'kalman': []}

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

        # Update adaptive weight controller
        weight_controller.update_errors(v1_error, kalman_error)

        errors['v1'].append(v1_error)
        errors['v2'].append(v2_error)
        errors['v3'].append(v3_error)
        errors['kalman'].append(kalman_error)

        prev_pos = pos

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/synthetic')
    parser.add_argument('--n_trials', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    trajectory_types = [
        ("linear", "直線（ベースライン）"),
        ("curved", "曲線（滑らか）"),
        ("sharp_turn", "急な90°ターン"),
        ("zigzag", "ジグザグ"),
        ("sudden_stop", "急停止→逆方向"),
        ("random_walk", "ランダムウォーク"),
    ]

    print("=" * 70)
    print("Synthetic Trajectory Test: V1/V2/V3/Kalman Comparison")
    print("=" * 70)

    all_results = {}

    for traj_type, traj_name in trajectory_types:
        print(f"\n### {traj_name} ({traj_type}) ###")

        v1_all, v2_all, v3_all, kalman_all = [], [], [], []

        for trial in range(args.n_trials):
            np.random.seed(trial * 100)
            trajectory = generate_synthetic_trajectory(traj_type, n_points=200)
            errors = evaluate_all_methods(trajectory, seed=trial)

            if errors and len(errors['v1']) > 0:
                v1_all.append(np.mean(errors['v1']))
                v2_all.append(np.mean(errors['v2']))
                v3_all.append(np.mean(errors['v3']))
                kalman_all.append(np.mean(errors['kalman']))
                print(f"  Trial {trial+1}: Kalman={np.mean(errors['kalman']):.4f}, V3={np.mean(errors['v3']):.4f}, V2={np.mean(errors['v2']):.4f}, V1={np.mean(errors['v1']):.4f}")

        if len(v1_all) > 0:
            all_results[traj_type] = {
                'name': traj_name,
                'v1': np.mean(v1_all),
                'v2': np.mean(v2_all),
                'v3': np.mean(v3_all),
                'kalman': np.mean(kalman_all)
            }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Trajectory':<20} {'Kalman':<10} {'V3':<10} {'V2':<10} {'V1':<10} {'Best':<10}")
    print("-" * 70)

    for traj_type, data in all_results.items():
        methods = {'Kalman': data['kalman'], 'V3': data['v3'], 'V2': data['v2'], 'V1': data['v1']}
        best = min(methods, key=methods.get)
        print(f"{data['name']:<20} {data['kalman']:<10.4f} {data['v3']:<10.4f} {data['v2']:<10.4f} {data['v1']:<10.4f} {best:<10}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (traj_type, data) in enumerate(all_results.items()):
        if idx >= 6:
            break
        ax = axes[idx]
        methods = ['Kalman', 'V3', 'V2', 'V1']
        values = [data['kalman'], data['v3'], data['v2'], data['v1']]
        colors = ['green' if v == min(values) else 'steelblue' for v in values]

        bars = ax.bar(methods, values, color=colors)
        ax.set_title(data['name'])
        ax.set_ylabel('Mean Error (m)')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Synthetic Trajectory: V1/V2/V3/Kalman Comparison', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f'synthetic_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")


if __name__ == "__main__":
    main()
