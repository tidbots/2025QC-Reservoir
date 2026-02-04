#!/usr/bin/env python3
"""
ETH Dataset: V3 Adaptive ESN Hybrid
Adaptive weighting based on trajectory complexity:
- Complex trajectory (direction changes, speed variation) -> ESN-heavy
- Simple trajectory (linear) -> Kalman-heavy
- Dynamic weight adjustment based on prediction errors
"""
import sys
import os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.10/site-packages'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    def get_velocity(self):
        return self.x[2:4]


class TrajectoryComplexityAnalyzer:
    """Analyze trajectory complexity to determine ESN vs Kalman weighting."""

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

                if norm1 > 1e-6 and norm2 > 1e-6:
                    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
                    angle = np.arccos(cos_angle)
                    self.direction_history.append(angle)

    def get_complexity_score(self):
        """
        Returns complexity score [0, 1]:
        0 = simple linear trajectory (Kalman preferred)
        1 = complex non-linear trajectory (ESN preferred)
        """
        if len(self.velocity_history) < 3:
            return 0.5  # Unknown, use balanced

        # Direction change score (0-1)
        if len(self.direction_history) > 0:
            avg_angle = np.mean(self.direction_history)
            max_angle = np.max(self.direction_history) if len(self.direction_history) > 0 else 0
            # Normalize: 0 rad = straight, pi rad = reversal
            direction_score = min(1.0, (avg_angle + max_angle * 0.5) / (np.pi * 0.5))
        else:
            direction_score = 0

        # Speed variation score (0-1)
        speeds = [np.linalg.norm(v) for v in self.velocity_history]
        if len(speeds) > 1 and np.mean(speeds) > 1e-6:
            speed_cv = np.std(speeds) / (np.mean(speeds) + 1e-6)  # Coefficient of variation
            speed_score = min(1.0, speed_cv * 2)  # Normalize
        else:
            speed_score = 0

        # Combined complexity score
        complexity = 0.6 * direction_score + 0.4 * speed_score
        return np.clip(complexity, 0, 1)


class AdaptiveWeightController:
    """Dynamically adjust ESN vs Kalman weight based on performance."""

    def __init__(self, initial_esn_weight=0.5, learning_rate=0.1):
        self.esn_weight = initial_esn_weight
        self.learning_rate = learning_rate
        self.esn_errors = deque(maxlen=20)
        self.kalman_errors = deque(maxlen=20)

    def update_errors(self, esn_error, kalman_error):
        self.esn_errors.append(esn_error)
        self.kalman_errors.append(kalman_error)

    def get_adaptive_weight(self, complexity_score):
        """
        Get ESN weight based on:
        1. Trajectory complexity (higher = more ESN)
        2. Recent prediction performance (lower error = higher weight)
        """
        # Base weight from complexity
        base_esn_weight = 0.3 + 0.5 * complexity_score  # Range: 0.3 - 0.8

        # Adjust based on recent errors
        if len(self.esn_errors) >= 5 and len(self.kalman_errors) >= 5:
            avg_esn_err = np.mean(self.esn_errors)
            avg_kalman_err = np.mean(self.kalman_errors)

            if avg_esn_err + avg_kalman_err > 1e-6:
                # Performance ratio: lower ESN error = higher ESN weight
                perf_ratio = avg_kalman_err / (avg_esn_err + avg_kalman_err)
                # Blend base weight with performance-based weight
                adaptive_weight = 0.7 * base_esn_weight + 0.3 * perf_ratio
            else:
                adaptive_weight = base_esn_weight
        else:
            adaptive_weight = base_esn_weight

        return np.clip(adaptive_weight, 0.2, 0.9)


def load_eth_dataset(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["frame", "ped_id", "x", "y"]
    return df


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


def evaluate_v3_adaptive(traj, warmup=5, window=20, n_models=10, future_horizon=20, seed=42):
    """V3: Adaptive ESN-Kalman hybrid based on trajectory complexity."""
    if len(traj) < warmup + future_horizon + 10:
        return None

    # Initialize components
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
    errors = {'v1_esn': [], 'v2_kalman_hybrid': [], 'v3_adaptive': []}
    esn_weights_used = []
    complexity_scores = []

    window_buffer = list(warmup_buffer)
    prev_pos = warmup_buffer[-1] if len(warmup_buffer) > 0 else None

    for frame_idx in range(warmup, len(traj) - future_horizon):
        pos = traj[frame_idx]
        window_buffer.append(pos)
        if len(window_buffer) > window:
            window_buffer.pop(0)

        # Update analyzers
        kalman.update(pos)
        complexity_analyzer.update(pos, prev_pos)
        in_online_std.update(pos.reshape(1, -1))

        # Get complexity score
        complexity = complexity_analyzer.get_complexity_score()
        complexity_scores.append(complexity)

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

        # V2: Fixed weight hybrid (kalman_weight=0.3)
        v2_pred = 0.7 * esn_pred + 0.3 * kalman_pred

        # V3: Adaptive weight based on complexity and performance
        esn_weight = weight_controller.get_adaptive_weight(complexity)
        v3_pred = esn_weight * esn_pred + (1 - esn_weight) * kalman_pred
        esn_weights_used.append(esn_weight)

        # Ground truth
        gt_future = traj[frame_idx + 1:frame_idx + 1 + future_horizon]

        if len(gt_future) >= future_horizon:
            esn_err = np.mean(np.linalg.norm(esn_pred - gt_future, axis=1))
            kalman_err = np.mean(np.linalg.norm(kalman_pred - gt_future, axis=1))
            v2_err = np.mean(np.linalg.norm(v2_pred - gt_future, axis=1))
            v3_err = np.mean(np.linalg.norm(v3_pred - gt_future, axis=1))

            errors['v1_esn'].append(esn_err)
            errors['v2_kalman_hybrid'].append(v2_err)
            errors['v3_adaptive'].append(v3_err)

            # Update weight controller with actual errors
            weight_controller.update_errors(esn_err, kalman_err)

        prev_pos = pos

    return {
        'errors': {k: np.array(v) for k, v in errors.items()},
        'esn_weights': np.array(esn_weights_used),
        'complexity_scores': np.array(complexity_scores)
    }


def main():
    parser = argparse.ArgumentParser(description='ETH V3 Adaptive ESN Evaluation')
    parser.add_argument('--data', default='data/students001_train.txt', help='Dataset path')
    parser.add_argument('--output', default='../output', help='Output directory')
    parser.add_argument('--ped_ids', type=int, nargs='+', default=[399, 168, 269, 177, 178])
    parser.add_argument('--n_models', type=int, default=10)
    parser.add_argument('--future_horizon', type=int, default=20)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data)
    output_dir = os.path.join(script_dir, args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("ETH Dataset: V3 Adaptive ESN Hybrid")
    print("="*70)
    print("V1: ESN only")
    print("V2: ESN + Kalman (fixed weight 0.7/0.3)")
    print("V3: ESN + Kalman (adaptive weight based on trajectory complexity)")
    print("="*70)

    df = load_eth_dataset(data_path)

    all_results = []
    for pid in args.ped_ids:
        print(f"\nPedestrian {pid}:")
        traj = df[df["ped_id"] == pid][["x", "y"]].to_numpy()

        if len(traj) < 50:
            print(f"  Skipping: insufficient data")
            continue

        print(f"  Evaluating ({len(traj)} frames)...")
        result = evaluate_v3_adaptive(traj, n_models=args.n_models,
                                      future_horizon=args.future_horizon)

        if result:
            v1_mean = np.mean(result['errors']['v1_esn'])
            v2_mean = np.mean(result['errors']['v2_kalman_hybrid'])
            v3_mean = np.mean(result['errors']['v3_adaptive'])
            avg_esn_weight = np.mean(result['esn_weights'])
            avg_complexity = np.mean(result['complexity_scores'])

            all_results.append({
                'ped_id': pid,
                'v1': v1_mean,
                'v2': v2_mean,
                'v3': v3_mean,
                'avg_esn_weight': avg_esn_weight,
                'avg_complexity': avg_complexity,
                'n_frames': len(result['errors']['v1_esn']),
                'raw': result
            })

            v3_vs_v1 = (v1_mean - v3_mean) / v1_mean * 100
            v3_vs_v2 = (v2_mean - v3_mean) / v2_mean * 100

            print(f"  V1 (ESN):        {v1_mean:.4f}m")
            print(f"  V2 (Fixed):      {v2_mean:.4f}m")
            print(f"  V3 (Adaptive):   {v3_mean:.4f}m (vs V1: {v3_vs_v1:+.1f}%, vs V2: {v3_vs_v2:+.1f}%)")
            print(f"  Avg ESN weight:  {avg_esn_weight:.2f}")
            print(f"  Avg complexity:  {avg_complexity:.2f}")

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        print(f"{'Ped ID':<10} {'V1 (ESN)':<12} {'V2 (Fixed)':<12} {'V3 (Adapt)':<12} {'ESN Wt':<10} {'Complx':<10}")
        print('-'*66)

        for r in all_results:
            print(f"{int(r['ped_id']):<10} {r['v1']:<12.4f} {r['v2']:<12.4f} {r['v3']:<12.4f} {r['avg_esn_weight']:<10.2f} {r['avg_complexity']:<10.2f}")

        print('-'*66)
        avg_v1 = np.mean([r['v1'] for r in all_results])
        avg_v2 = np.mean([r['v2'] for r in all_results])
        avg_v3 = np.mean([r['v3'] for r in all_results])
        avg_wt = np.mean([r['avg_esn_weight'] for r in all_results])
        avg_cx = np.mean([r['avg_complexity'] for r in all_results])

        v3_vs_v1 = (avg_v1 - avg_v3) / avg_v1 * 100
        v3_vs_v2 = (avg_v2 - avg_v3) / avg_v2 * 100

        print(f"{'Average':<10} {avg_v1:<12.4f} {avg_v2:<12.4f} {avg_v3:<12.4f} {avg_wt:<10.2f} {avg_cx:<10.2f}")
        print(f"\nV3 vs V1: {v3_vs_v1:+.1f}%")
        print(f"V3 vs V2: {v3_vs_v2:+.1f}%")
        print('='*70)

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Bar comparison
        ax = axes[0, 0]
        x = np.arange(len(all_results))
        width = 0.25
        ax.bar(x - width, [r['v1'] for r in all_results], width, label='V1 (ESN)', color='steelblue', alpha=0.8)
        ax.bar(x, [r['v2'] for r in all_results], width, label='V2 (Fixed)', color='forestgreen', alpha=0.8)
        ax.bar(x + width, [r['v3'] for r in all_results], width, label='V3 (Adaptive)', color='darkred', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Ped {int(r['ped_id'])}" for r in all_results], rotation=45)
        ax.set_ylabel('Mean Error (m)')
        ax.set_title('V1 vs V2 vs V3 by Pedestrian')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Overall comparison
        ax = axes[0, 1]
        methods = ['V1\n(ESN)', 'V2\n(Fixed)', 'V3\n(Adaptive)']
        values = [avg_v1, avg_v2, avg_v3]
        colors = ['steelblue', 'forestgreen', 'darkred']
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        ax.set_ylabel('Average Error (m)')
        ax.set_title(f'Overall Comparison\nV3 vs V1: {v3_vs_v1:+.1f}%, V3 vs V2: {v3_vs_v2:+.1f}%')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # ESN weight distribution
        ax = axes[1, 0]
        all_weights = np.concatenate([r['raw']['esn_weights'] for r in all_results])
        ax.hist(all_weights, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(all_weights), color='red', linestyle='--',
                  label=f'Mean: {np.mean(all_weights):.2f}')
        ax.axvline(x=0.7, color='green', linestyle='--', label='V2 fixed: 0.70')
        ax.set_xlabel('ESN Weight')
        ax.set_ylabel('Frequency')
        ax.set_title('Adaptive ESN Weight Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Complexity vs ESN weight scatter
        ax = axes[1, 1]
        for r in all_results:
            ax.scatter(r['raw']['complexity_scores'], r['raw']['esn_weights'],
                      alpha=0.3, s=10, label=f"Ped {int(r['ped_id'])}")
        ax.set_xlabel('Trajectory Complexity')
        ax.set_ylabel('ESN Weight')
        ax.set_title('Complexity vs ESN Weight')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.suptitle('ETH Dataset: V3 Adaptive ESN Hybrid\n'
                    f'{len(all_results)} Pedestrians, {args.future_horizon}-step Horizon',
                    fontsize=14)
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"eth_v3_adaptive_{timestamp}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved: {out_path}")


if __name__ == "__main__":
    main()
