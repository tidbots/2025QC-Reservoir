#!/usr/bin/env python3
"""
Disturbance Response Test: ESN vs Kalman
Evaluate prediction performance under sudden, unpredictable disturbances.
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
        sr = float(rng.uniform(0.90, 0.98))
        lr = float(rng.uniform(0.50, 0.70))
        input_scaling = float(rng.uniform(0.40, 0.60))
        bias = float(rng.uniform(-0.1, 0.1))
        reservoir = Reservoir(units=units, sr=sr, lr=lr, input_scaling=input_scaling,
                             bias=bias, seed=int(seed + i))
        readout = RLS(forgetting=rls_forgetting)
        esn = reservoir >> readout
        esns.append(esn)
    return esns


def generate_disturbance_trajectory(disturbance_type, n_points=300, noise=0.01, seed=42):
    """
    Generate trajectory with sudden, unpredictable disturbances.
    Returns: trajectory, disturbance_frames (list of frame indices where disturbance occurs)
    """
    np.random.seed(seed)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    disturbance_frames = []

    if disturbance_type == "sudden_obstacle":
        # Walking straight, then sudden 90 degree turn due to obstacle
        # Multiple random obstacles
        vx, vy = 0.05, 0.02
        obstacle_times = sorted(np.random.choice(range(50, n_points-50), size=3, replace=False))

        for i in range(1, n_points):
            if i in obstacle_times:
                # Sudden 90 degree turn
                vx, vy = -vy, vx  # Rotate 90 degrees
                disturbance_frames.append(i)
            x[i] = x[i-1] + vx
            y[i] = y[i-1] + vy

    elif disturbance_type == "random_push":
        # Walking with random push disturbances
        vx, vy = 0.05, 0.02
        push_times = sorted(np.random.choice(range(30, n_points-30), size=5, replace=False))

        for i in range(1, n_points):
            if i in push_times:
                # Random push
                push_angle = np.random.uniform(0, 2*np.pi)
                push_strength = np.random.uniform(0.3, 0.8)
                vx += push_strength * np.cos(push_angle)
                vy += push_strength * np.sin(push_angle)
                disturbance_frames.append(i)
            else:
                # Gradual return to original direction
                vx = 0.9 * vx + 0.1 * 0.05
                vy = 0.9 * vy + 0.1 * 0.02
            x[i] = x[i-1] + vx
            y[i] = y[i-1] + vy

    elif disturbance_type == "goal_switch":
        # Walking toward goal, then goal suddenly changes
        goals = [(10, 5), (-5, 8), (8, -3), (0, 10)]
        goal_idx = 0
        switch_times = sorted(np.random.choice(range(40, n_points-40), size=3, replace=False))

        for i in range(1, n_points):
            if i in switch_times:
                goal_idx = (goal_idx + 1) % len(goals)
                disturbance_frames.append(i)

            # Move toward current goal
            goal = goals[goal_idx]
            dx = goal[0] - x[i-1]
            dy = goal[1] - y[i-1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.01:
                speed = 0.05
                x[i] = x[i-1] + speed * dx / dist
                y[i] = y[i-1] + speed * dy / dist
            else:
                x[i] = x[i-1]
                y[i] = y[i-1]

    elif disturbance_type == "sudden_stop_reverse":
        # Walking, sudden stop, then reverse
        vx, vy = 0.05, 0.02
        events = sorted(np.random.choice(range(50, n_points-50), size=4, replace=False))
        state = "forward"

        for i in range(1, n_points):
            if i in events:
                if state == "forward":
                    state = "stopped"
                    vx, vy = 0, 0
                elif state == "stopped":
                    state = "reverse"
                    vx, vy = -0.05, -0.02
                else:
                    state = "forward"
                    vx, vy = 0.05, 0.02
                disturbance_frames.append(i)

            x[i] = x[i-1] + vx
            y[i] = y[i-1] + vy

    elif disturbance_type == "speed_burst":
        # Normal walking with sudden speed bursts
        base_speed = 0.05
        angle = 0.3
        burst_times = sorted(np.random.choice(range(30, n_points-30), size=5, replace=False))
        burst_duration = 10

        for i in range(1, n_points):
            speed = base_speed

            # Check if in burst period
            for bt in burst_times:
                if bt <= i < bt + burst_duration:
                    speed = base_speed * 3  # Triple speed
                    if i == bt:
                        disturbance_frames.append(i)
                    break

            x[i] = x[i-1] + speed * np.cos(angle)
            y[i] = y[i-1] + speed * np.sin(angle)

    elif disturbance_type == "erratic_motion":
        # Smooth motion with erratic bursts
        t = np.linspace(0, 4*np.pi, n_points)
        x = t * 0.5
        y = np.sin(t) * 2

        # Add erratic bursts
        burst_times = sorted(np.random.choice(range(30, n_points-30), size=4, replace=False))
        for bt in burst_times:
            disturbance_frames.append(bt)
            burst_len = min(15, n_points - bt)
            for j in range(burst_len):
                x[bt+j] += np.random.randn() * 0.3
                y[bt+j] += np.random.randn() * 0.3

    # Add noise
    x += np.random.randn(n_points) * noise
    y += np.random.randn(n_points) * noise

    return np.column_stack([x, y]), disturbance_frames


def evaluate_at_disturbance(traj, disturbance_frames, warmup=10, window=20,
                            n_models=10, future_horizon=20, seed=42,
                            eval_window=15):
    """
    Evaluate prediction error specifically around disturbance events.
    eval_window: frames before/after disturbance to evaluate
    """
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

    # Storage - separate for disturbance and normal periods
    errors_disturb = {'v1': [], 'v2': [], 'v3': [], 'kalman': [], 'linear': []}
    errors_normal = {'v1': [], 'v2': [], 'v3': [], 'kalman': [], 'linear': []}

    # Determine disturbance evaluation frames
    disturb_eval_frames = set()
    for df in disturbance_frames:
        for offset in range(-eval_window, eval_window + future_horizon):
            disturb_eval_frames.add(df + offset)

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

        # Calculate errors
        v1_final = esn_pred[-1]
        v1_error = np.linalg.norm(v1_final - gt_pos)

        v2_final = 0.7 * esn_pred[-1] + 0.3 * kalman_pred[-1]
        v2_error = np.linalg.norm(v2_final - gt_pos)

        esn_weight = weight_controller.get_adaptive_weight(complexity)
        v3_final = esn_weight * esn_pred[-1] + (1 - esn_weight) * kalman_pred[-1]
        v3_error = np.linalg.norm(v3_final - gt_pos)

        kalman_final = kalman_pred[-1]
        kalman_error = np.linalg.norm(kalman_final - gt_pos)

        linear_error = np.linalg.norm(linear_pred - gt_pos)

        weight_controller.update_errors(v1_error, kalman_error)

        # Store in appropriate category
        if frame_idx in disturb_eval_frames:
            errors_disturb['v1'].append(v1_error)
            errors_disturb['v2'].append(v2_error)
            errors_disturb['v3'].append(v3_error)
            errors_disturb['kalman'].append(kalman_error)
            errors_disturb['linear'].append(linear_error)
        else:
            errors_normal['v1'].append(v1_error)
            errors_normal['v2'].append(v2_error)
            errors_normal['v3'].append(v3_error)
            errors_normal['kalman'].append(kalman_error)
            errors_normal['linear'].append(linear_error)

        prev_pos = pos

    return errors_disturb, errors_normal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/disturbance')
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--future_horizon', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    disturbance_types = [
        ("sudden_obstacle", "突発障害物"),
        ("random_push", "ランダム外力"),
        ("goal_switch", "目標切替"),
        ("sudden_stop_reverse", "急停止・反転"),
        ("speed_burst", "速度バースト"),
        ("erratic_motion", "不規則動作"),
    ]

    print("=" * 80)
    print("Disturbance Response Test: ESN vs Kalman")
    print("Evaluating prediction under unpredictable disturbances")
    print(f"Future Horizon: {args.future_horizon} steps")
    print("=" * 80)

    all_results = {}

    for dist_type, dist_name in disturbance_types:
        print(f"\n### {dist_name} ({dist_type}) ###")

        disturb_v1, disturb_v3, disturb_kalman = [], [], []
        normal_v1, normal_v3, normal_kalman = [], [], []

        for trial in range(args.n_trials):
            traj, dist_frames = generate_disturbance_trajectory(
                dist_type, n_points=300, seed=trial*100
            )

            result = evaluate_at_disturbance(
                traj, dist_frames, seed=trial, future_horizon=args.future_horizon
            )

            if result:
                errors_disturb, errors_normal = result

                if len(errors_disturb['v1']) > 0:
                    disturb_v1.append(np.mean(errors_disturb['v1']))
                    disturb_v3.append(np.mean(errors_disturb['v3']))
                    disturb_kalman.append(np.mean(errors_disturb['kalman']))

                if len(errors_normal['v1']) > 0:
                    normal_v1.append(np.mean(errors_normal['v1']))
                    normal_v3.append(np.mean(errors_normal['v3']))
                    normal_kalman.append(np.mean(errors_normal['kalman']))

        if disturb_kalman:
            all_results[dist_type] = {
                'name': dist_name,
                'disturb': {
                    'v1': np.mean(disturb_v1),
                    'v3': np.mean(disturb_v3),
                    'kalman': np.mean(disturb_kalman),
                },
                'normal': {
                    'v1': np.mean(normal_v1) if normal_v1 else np.nan,
                    'v3': np.mean(normal_v3) if normal_v3 else np.nan,
                    'kalman': np.mean(normal_kalman) if normal_kalman else np.nan,
                }
            }

            print(f"  Disturbance: Kalman={all_results[dist_type]['disturb']['kalman']:.3f}, "
                  f"V3={all_results[dist_type]['disturb']['v3']:.3f}, "
                  f"V1={all_results[dist_type]['disturb']['v1']:.3f}")
            print(f"  Normal:      Kalman={all_results[dist_type]['normal']['kalman']:.3f}, "
                  f"V3={all_results[dist_type]['normal']['v3']:.3f}, "
                  f"V1={all_results[dist_type]['normal']['v1']:.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Disturbance Response Results")
    print("=" * 80)

    print("\n### Disturbance Period (around disturbance events) ###")
    header = "Disturbance Type     Kalman     V3         V1         Best       V3 vs Kalman"
    print(header)
    print("-" * 80)

    esn_wins_disturb = 0

    for dist_type, data in all_results.items():
        d = data['disturb']
        methods = {'Kalman': d['kalman'], 'V3': d['v3'], 'V1': d['v1']}
        best = min(methods, key=methods.get)

        if best in ['V1', 'V3']:
            esn_wins_disturb += 1

        diff = (d['kalman'] - d['v3']) / d['kalman'] * 100 if d['kalman'] > 0 else 0
        status = "BETTER" if diff > 0 else "worse"

        print(f"{data['name']:<20} {d['kalman']:<10.3f} {d['v3']:<10.3f} "
              f"{d['v1']:<10.3f} {best:<10} {diff:+.1f}% ({status})")

    print("\n### Normal Period (stable motion) ###")
    print(header)
    print("-" * 80)

    for dist_type, data in all_results.items():
        n = data['normal']
        if np.isnan(n['kalman']):
            continue
        methods = {'Kalman': n['kalman'], 'V3': n['v3'], 'V1': n['v1']}
        best = min(methods, key=methods.get)

        diff = (n['kalman'] - n['v3']) / n['kalman'] * 100 if n['kalman'] > 0 else 0
        status = "BETTER" if diff > 0 else "worse"

        print(f"{data['name']:<20} {n['kalman']:<10.3f} {n['v3']:<10.3f} "
              f"{n['v1']:<10.3f} {best:<10} {diff:+.1f}% ({status})")

    print("\n" + "=" * 80)
    print(f"ESN (V1/V3) wins during disturbance: {esn_wins_disturb}/{len(all_results)}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (dist_type, data) in enumerate(all_results.items()):
        if idx >= 6:
            break
        ax = axes[idx]

        # Bar chart comparing disturbance vs normal
        x = np.arange(3)
        width = 0.35

        disturb_vals = [data['disturb']['kalman'], data['disturb']['v3'], data['disturb']['v1']]
        normal_vals = [data['normal']['kalman'], data['normal']['v3'], data['normal']['v1']]

        bars1 = ax.bar(x - width/2, disturb_vals, width, label='Disturbance', color='coral')
        bars2 = ax.bar(x + width/2, normal_vals, width, label='Normal', color='steelblue')

        ax.set_ylabel('Mean Error')
        ax.set_title(data['name'])
        ax.set_xticks(x)
        ax.set_xticklabels(['Kalman', 'V3', 'V1'])
        ax.legend()

        # Highlight best in disturbance period
        min_disturb = min(disturb_vals)
        for bar, val in zip(bars1, disturb_vals):
            if val == min_disturb:
                bar.set_edgecolor('green')
                bar.set_linewidth(2)

    plt.suptitle('Disturbance Response: ESN vs Kalman Comparison', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f'disturbance_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")

    # Trajectory visualization with disturbance markers
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()

    for idx, (dist_type, dist_name) in enumerate(disturbance_types):
        if idx >= 6:
            break
        ax = axes2[idx]

        traj, dist_frames = generate_disturbance_trajectory(dist_type, n_points=300, seed=42)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.7)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'ko', markersize=8, label='End')

        # Mark disturbance points
        for df in dist_frames:
            if df < len(traj):
                ax.plot(traj[df, 0], traj[df, 1], 'r*', markersize=12)

        ax.plot([], [], 'r*', markersize=12, label='Disturbance')
        ax.set_title(dist_name)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Disturbance Trajectory Patterns', fontsize=14)
    plt.tight_layout()

    traj_path = os.path.join(args.output, f'disturbance_trajectories_{timestamp}.png')
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory patterns saved: {traj_path}")


if __name__ == "__main__":
    main()
