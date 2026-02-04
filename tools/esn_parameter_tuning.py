#!/usr/bin/env python3
"""
ESN Parameter Tuning: Grid search for optimal parameters
Focus on improving performance during direction changes.
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
from itertools import product


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


def create_esns_with_params(n_models, units, sr, lr, input_scaling, rls_forgetting, seed=42):
    """Create ESN ensemble with specified parameters."""
    esns = []
    for i in range(n_models):
        reservoir = Reservoir(
            units=units,
            sr=sr,
            lr=lr,
            input_scaling=input_scaling,
            bias=0.0,
            seed=int(seed + i)
        )
        readout = RLS(forgetting=rls_forgetting)
        esn = reservoir >> readout
        esns.append(esn)
    return esns


def find_direction_change_frames(trajectory, angle_threshold=20):
    """Find frames where direction changes more than threshold degrees."""
    direction_change_frames = []
    velocities = np.diff(trajectory, axis=0)

    for i in range(len(velocities) - 1):
        v1 = velocities[i]
        v2 = velocities[i + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

        if n1 > 0.01 and n2 > 0.01:
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angle = np.degrees(np.arccos(cos_angle))

            if angle > angle_threshold:
                direction_change_frames.append(i + 1)

    return direction_change_frames


def evaluate_with_params(traj, params, warmup=5, window=20, n_models=5, future_horizon=20, seed=42):
    """Evaluate ESN with specific parameters."""

    if len(traj) < warmup + future_horizon + 10:
        return None

    # Find direction change frames
    dc_frames = find_direction_change_frames(traj, angle_threshold=20)
    valid_dc_frames = [f for f in dc_frames if f >= warmup and f < len(traj) - future_horizon]

    # Create ESNs with given parameters
    esns = create_esns_with_params(
        n_models=n_models,
        units=params['units'],
        sr=params['sr'],
        lr=params['lr'],
        input_scaling=params['input_scaling'],
        rls_forgetting=params['rls_forgetting'],
        seed=seed
    )

    kalman = SimpleKalmanFilter(dt=0.1)

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

    # Config
    adapt_window = 5
    delta_window = 6
    adapt_damping = params.get('adapt_damping', 0.35)
    err_clip = 1.0

    # Results
    dc_esn_errors = []
    dc_kalman_errors = []
    non_dc_esn_errors = []
    non_dc_kalman_errors = []

    window_buffer = list(warmup_buffer)

    for frame_idx in range(warmup, len(traj) - future_horizon):
        pos = traj[frame_idx]
        window_buffer.append(pos)
        if len(window_buffer) > window:
            window_buffer.pop(0)

        kalman.update(pos)
        in_online_std.update(pos.reshape(1, -1))

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

        esn_error = np.linalg.norm(esn_pred[-1] - gt_pos)
        kalman_error = np.linalg.norm(kalman_pred[-1] - gt_pos)

        is_dc = frame_idx in valid_dc_frames
        if is_dc:
            dc_esn_errors.append(esn_error)
            dc_kalman_errors.append(kalman_error)
        else:
            non_dc_esn_errors.append(esn_error)
            non_dc_kalman_errors.append(kalman_error)

    # Overall errors
    all_esn_errors = dc_esn_errors + non_dc_esn_errors
    all_kalman_errors = dc_kalman_errors + non_dc_kalman_errors

    return {
        'dc_esn': np.mean(dc_esn_errors) if dc_esn_errors else None,
        'dc_kalman': np.mean(dc_kalman_errors) if dc_kalman_errors else None,
        'non_dc_esn': np.mean(non_dc_esn_errors) if non_dc_esn_errors else None,
        'non_dc_kalman': np.mean(non_dc_kalman_errors) if non_dc_kalman_errors else None,
        'overall_esn': np.mean(all_esn_errors) if all_esn_errors else None,
        'overall_kalman': np.mean(all_kalman_errors) if all_kalman_errors else None,
        'n_dc': len(dc_esn_errors),
        'n_non_dc': len(non_dc_esn_errors)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/students001_train.txt')
    parser.add_argument('--output', default='output/tuning')
    parser.add_argument('--n_peds', type=int, default=10)
    parser.add_argument('--quick', action='store_true', help='Quick search with fewer parameters')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    os.makedirs(args.output, exist_ok=True)

    # Load data
    data = pd.read_csv(args.data, sep='\s+', header=None, names=['frame', 'ped_id', 'x', 'y'])

    # Get pedestrians
    ped_frames = data.groupby('ped_id').size()
    valid_peds = ped_frames[ped_frames >= 80].index.tolist()[:args.n_peds]

    print("=" * 70)
    print("ESN Parameter Tuning")
    print("=" * 70)

    # Parameter grid
    if args.quick:
        param_grid = {
            'units': [20, 35, 50],
            'sr': [0.8, 0.95],
            'lr': [0.3, 0.6],
            'input_scaling': [0.3, 0.5],
            'rls_forgetting': [0.95, 0.99],
        }
    else:
        param_grid = {
            'units': [15, 25, 35, 50],
            'sr': [0.7, 0.85, 0.95],
            'lr': [0.2, 0.4, 0.6],
            'input_scaling': [0.2, 0.4, 0.6],
            'rls_forgetting': [0.9, 0.95, 0.99],
        }

    # Current best params (baseline)
    baseline_params = {
        'units': 25,
        'sr': 0.85,
        'lr': 0.45,
        'input_scaling': 0.3,
        'rls_forgetting': 0.99,
    }

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    print(f"Testing {len(all_combinations)} parameter combinations on {len(valid_peds)} pedestrians")
    print(f"Baseline: {baseline_params}")

    results = []

    for idx, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))

        overall_esn_errors = []
        overall_kalman_errors = []
        dc_esn_errors = []
        dc_kalman_errors = []

        for ped_id in valid_peds:
            ped_data = data[data['ped_id'] == ped_id].sort_values('frame')
            trajectory = ped_data[['x', 'y']].values

            result = evaluate_with_params(trajectory, params, seed=int(ped_id))

            if result and result['overall_esn'] is not None:
                overall_esn_errors.append(result['overall_esn'])
                overall_kalman_errors.append(result['overall_kalman'])
                if result['dc_esn'] is not None:
                    dc_esn_errors.append(result['dc_esn'])
                    dc_kalman_errors.append(result['dc_kalman'])

        if len(overall_esn_errors) > 0:
            avg_overall_esn = np.mean(overall_esn_errors)
            avg_overall_kalman = np.mean(overall_kalman_errors)
            overall_improvement = (avg_overall_kalman - avg_overall_esn) / avg_overall_kalman * 100

            avg_dc_esn = np.mean(dc_esn_errors) if dc_esn_errors else None
            avg_dc_kalman = np.mean(dc_kalman_errors) if dc_kalman_errors else None
            dc_improvement = ((avg_dc_kalman - avg_dc_esn) / avg_dc_kalman * 100) if avg_dc_esn else None

            results.append({
                **params,
                'overall_esn': avg_overall_esn,
                'overall_kalman': avg_overall_kalman,
                'overall_improvement': overall_improvement,
                'dc_esn': avg_dc_esn,
                'dc_kalman': avg_dc_kalman,
                'dc_improvement': dc_improvement
            })

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[{idx+1}/{len(all_combinations)}] units={params['units']}, sr={params['sr']:.2f}, "
                      f"lr={params['lr']:.1f} -> Overall ESN: {avg_overall_esn:.3f}m, Kalman: {avg_overall_kalman:.3f}m, "
                      f"Improvement: {overall_improvement:+.1f}%")

    # Sort by overall improvement (higher is better = lower ESN error)
    results = sorted(results, key=lambda x: x['overall_esn'])

    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS (by overall ESN error - lower is better)")
    print("=" * 70)
    print(f"{'Units':<6} {'SR':<6} {'LR':<6} {'InpSc':<6} {'Forg':<6} {'ESN':<8} {'Kalman':<8} {'DC_Impr':<8}")
    print("-" * 70)

    for r in results[:10]:
        dc_impr_str = f"{r['dc_improvement']:+.1f}%" if r['dc_improvement'] else "N/A"
        print(f"{r['units']:<6} {r['sr']:<6.2f} {r['lr']:<6.2f} {r['input_scaling']:<6.2f} "
              f"{r['rls_forgetting']:<6.2f} {r['overall_esn']:<8.3f} {r['overall_kalman']:<8.3f} {dc_impr_str:>8}")

    # Best params
    best = results[0]
    print(f"\n{'='*70}")
    print("BEST PARAMETERS (lowest overall ESN error):")
    print(f"  units: {best['units']}")
    print(f"  spectral_radius: {best['sr']}")
    print(f"  leaking_rate: {best['lr']}")
    print(f"  input_scaling: {best['input_scaling']}")
    print(f"  rls_forgetting: {best['rls_forgetting']}")
    print(f"  Overall ESN Error: {best['overall_esn']:.3f}m")
    print(f"  Overall Kalman Error: {best['overall_kalman']:.3f}m")
    print(f"  Overall vs Kalman: {best['overall_improvement']:+.1f}%")
    if best['dc_improvement']:
        print(f"  Direction Change Improvement: {best['dc_improvement']:+.1f}%")

    # Evaluate baseline for comparison
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON:")
    baseline_overall = []
    baseline_kalman = []
    for ped_id in valid_peds:
        ped_data = data[data['ped_id'] == ped_id].sort_values('frame')
        trajectory = ped_data[['x', 'y']].values
        result = evaluate_with_params(trajectory, baseline_params, seed=int(ped_id))
        if result and result['overall_esn'] is not None:
            baseline_overall.append(result['overall_esn'])
            baseline_kalman.append(result['overall_kalman'])

    if baseline_overall:
        baseline_avg = np.mean(baseline_overall)
        baseline_kalman_avg = np.mean(baseline_kalman)
        baseline_impr = (baseline_kalman_avg - baseline_avg) / baseline_kalman_avg * 100
        print(f"  Baseline ESN Error: {baseline_avg:.3f}m")
        print(f"  Baseline vs Kalman: {baseline_impr:+.1f}%")
        print(f"  Best vs Baseline: {(baseline_avg - best['overall_esn']) / baseline_avg * 100:+.1f}%")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Units vs ESN Error
    ax = axes[0, 0]
    for units in param_grid['units']:
        subset = [r for r in results if r['units'] == units]
        if subset:
            ax.scatter([units] * len(subset), [r['overall_esn'] for r in subset], alpha=0.5)
    ax.set_xlabel('Units')
    ax.set_ylabel('Overall ESN Error (m)')
    ax.set_title('Reservoir Size vs ESN Error')

    # Plot 2: Spectral Radius vs ESN Error
    ax = axes[0, 1]
    for sr in param_grid['sr']:
        subset = [r for r in results if r['sr'] == sr]
        if subset:
            ax.scatter([sr] * len(subset), [r['overall_esn'] for r in subset], alpha=0.5)
    ax.set_xlabel('Spectral Radius')
    ax.set_ylabel('Overall ESN Error (m)')
    ax.set_title('Spectral Radius vs ESN Error')

    # Plot 3: Leaking Rate vs ESN Error
    ax = axes[1, 0]
    for lr in param_grid['lr']:
        subset = [r for r in results if r['lr'] == lr]
        if subset:
            ax.scatter([lr] * len(subset), [r['overall_esn'] for r in subset], alpha=0.5)
    ax.set_xlabel('Leaking Rate')
    ax.set_ylabel('Overall ESN Error (m)')
    ax.set_title('Leaking Rate vs ESN Error')

    # Plot 4: Top 10 comparison
    ax = axes[1, 1]
    top10 = results[:10]
    x = range(len(top10))
    kalman_avg = np.mean([r['overall_kalman'] for r in top10])
    ax.bar(x, [r['overall_esn'] for r in top10], color='steelblue', label='ESN')
    ax.axhline(y=kalman_avg, color='r', linestyle='--', alpha=0.7, label=f'Kalman ({kalman_avg:.3f}m)')
    ax.set_xlabel('Parameter Combination Rank')
    ax.set_ylabel('Overall ESN Error (m)')
    ax.set_title('Top 10 Parameter Combinations')
    ax.legend()

    plt.suptitle('ESN Parameter Tuning Results', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f'tuning_results_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")

    # Save results to CSV
    csv_path = os.path.join(args.output, f'tuning_results_{timestamp}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")


if __name__ == "__main__":
    main()
