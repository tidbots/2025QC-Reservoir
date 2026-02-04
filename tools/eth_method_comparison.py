#!/usr/bin/env python3
"""
ETH Dataset: ESN vs Conventional Methods Comparison
Compare ESN with traditional trajectory prediction methods:
- Linear extrapolation
- f(x) average (linear + parabola + sigmoid) - from RSJ2025 paper
- ESN ensemble
- ESN + Kalman hybrid
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
from scipy.optimize import curve_fit
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
    """Simple Kalman Filter for 2D position tracking."""
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


# ============================================
# Conventional Methods (from RSJ2025 paper)
# ============================================

def predict_linear(history, future_steps):
    """Linear extrapolation based on recent positions."""
    if len(history) < 2:
        return np.tile(history[-1], (future_steps, 1))

    xs = np.arange(len(history))
    lin_x = np.polyfit(xs, history[:, 0], 1)
    lin_y = np.polyfit(xs, history[:, 1], 1)

    x_future = np.arange(len(history), len(history) + future_steps)
    pred_x = np.polyval(lin_x, x_future)
    pred_y = np.polyval(lin_y, x_future)

    return np.column_stack((pred_x, pred_y))


def fit_sigmoid(x, y):
    """Fit sigmoid curve to data."""
    def sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    p0 = [max(y) - min(y), np.median(x), 1, min(y)]
    try:
        popt, _ = curve_fit(sigmoid, x, y, p0, maxfev=5000)
        return popt
    except:
        return p0


def predict_fx_average(history, future_steps):
    """
    f(x) = (l(x) + p(x) + s(x)) / 3
    Average of linear, parabolic, and sigmoid predictions.
    (Method from RSJ2025 paper: 1I5-03)
    """
    if len(history) < 3:
        return predict_linear(history, future_steps)

    xs = np.arange(len(history))

    # Linear fit
    lin_x = np.polyfit(xs, history[:, 0], 1)
    lin_y = np.polyfit(xs, history[:, 1], 1)

    # Parabolic fit
    par_x = np.polyfit(xs, history[:, 0], 2)
    par_y = np.polyfit(xs, history[:, 1], 2)

    # Sigmoid fit
    sig_popt_x = fit_sigmoid(xs, history[:, 0])
    sig_popt_y = fit_sigmoid(xs, history[:, 1])

    def sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    # Extrapolate
    x_future = np.arange(len(history), len(history) + future_steps)

    lin_pred_x = np.polyval(lin_x, x_future)
    lin_pred_y = np.polyval(lin_y, x_future)

    par_pred_x = np.polyval(par_x, x_future)
    par_pred_y = np.polyval(par_y, x_future)

    sig_pred_x = sigmoid(x_future, *sig_popt_x)
    sig_pred_y = sigmoid(x_future, *sig_popt_y)

    # Average: f(x) = (l(x) + p(x) + s(x)) / 3
    fx_pred_x = (lin_pred_x + par_pred_x + sig_pred_x) / 3
    fx_pred_y = (lin_pred_y + par_pred_y + sig_pred_y) / 3

    return np.column_stack((fx_pred_x, fx_pred_y))


def predict_kalman_only(kalman, future_steps):
    """Kalman filter prediction only."""
    return kalman.predict_future(future_steps)


# ============================================
# Evaluation
# ============================================

def evaluate_all_methods(traj, warmup=5, window=20, n_models=10, future_horizon=20, seed=42):
    """Evaluate all methods on a single trajectory."""
    if len(traj) < warmup + future_horizon + 10:
        return None

    # Initialize ESN
    esns = create_diverse_esns(n_models=n_models, base_units=25, seed=seed)
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

    for pos in warmup_buffer:
        kalman.update(pos)

    # Config
    adapt_window = 5
    delta_window = 6
    adapt_damping = 0.35
    err_clip = 1.0
    kalman_weight = 0.3

    # Error storage
    errors = {
        'linear': [],
        'fx_average': [],
        'kalman': [],
        'esn': [],
        'esn_kalman': []
    }

    window_buffer = list(warmup_buffer)

    for frame_idx in range(warmup, len(traj) - future_horizon):
        pos = traj[frame_idx]
        window_buffer.append(pos)
        if len(window_buffer) > window:
            window_buffer.pop(0)

        kalman.update(pos)
        in_online_std.update(pos.reshape(1, -1))

        history = np.array(window_buffer)

        # 1. Linear prediction
        linear_pred = predict_linear(history, future_horizon)

        # 2. f(x) average prediction (RSJ2025 paper method)
        fx_pred = predict_fx_average(history, future_horizon)

        # 3. Kalman only prediction
        kalman_pred = predict_kalman_only(kalman, future_horizon)

        # 4. ESN prediction
        X_recent = np.array(window_buffer[-adapt_window:])
        X_recent_s = in_online_std.transform(X_recent)

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
                last_pos = in_online_std.inverse_transform(last_input[-1].reshape(1, -1))[0]
                next_pos = last_pos + delta
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

        # 5. ESN + Kalman hybrid
        esn_kalman_pred = (1 - kalman_weight) * esn_pred + kalman_weight * kalman_pred

        # Ground truth
        gt_future = traj[frame_idx + 1:frame_idx + 1 + future_horizon]

        if len(gt_future) >= future_horizon:
            errors['linear'].append(np.mean(np.linalg.norm(linear_pred - gt_future, axis=1)))
            errors['fx_average'].append(np.mean(np.linalg.norm(fx_pred - gt_future, axis=1)))
            errors['kalman'].append(np.mean(np.linalg.norm(kalman_pred - gt_future, axis=1)))
            errors['esn'].append(np.mean(np.linalg.norm(esn_pred - gt_future, axis=1)))
            errors['esn_kalman'].append(np.mean(np.linalg.norm(esn_kalman_pred - gt_future, axis=1)))

    return {k: np.array(v) for k, v in errors.items()}


def main():
    parser = argparse.ArgumentParser(description='ETH: ESN vs Conventional Methods')
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
    print("ETH Dataset: ESN vs Conventional Methods Comparison")
    print("="*70)
    print("Methods:")
    print("  1. Linear - Linear extrapolation")
    print("  2. f(x)   - Linear + Parabola + Sigmoid average (RSJ2025 paper)")
    print("  3. Kalman - Kalman filter only")
    print("  4. ESN    - Echo State Network ensemble")
    print("  5. ESN+KF - ESN + Kalman hybrid (V2)")
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
        errors = evaluate_all_methods(traj, n_models=args.n_models,
                                      future_horizon=args.future_horizon)

        if errors:
            result = {
                'ped_id': pid,
                'linear': np.mean(errors['linear']),
                'fx_average': np.mean(errors['fx_average']),
                'kalman': np.mean(errors['kalman']),
                'esn': np.mean(errors['esn']),
                'esn_kalman': np.mean(errors['esn_kalman']),
                'n_frames': len(errors['esn'])
            }
            all_results.append(result)

            print(f"  Linear:     {result['linear']:.4f}m")
            print(f"  f(x):       {result['fx_average']:.4f}m")
            print(f"  Kalman:     {result['kalman']:.4f}m")
            print(f"  ESN:        {result['esn']:.4f}m")
            print(f"  ESN+Kalman: {result['esn_kalman']:.4f}m")

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        print(f"{'Method':<15} {'Mean Error (m)':<15} {'vs Linear':<12} {'vs f(x)':<12}")
        print('-'*54)

        methods = ['linear', 'fx_average', 'kalman', 'esn', 'esn_kalman']
        method_names = ['Linear', 'f(x) avg', 'Kalman', 'ESN', 'ESN+Kalman']

        avg_errors = {}
        for method in methods:
            avg_errors[method] = np.mean([r[method] for r in all_results])

        for method, name in zip(methods, method_names):
            vs_linear = (avg_errors['linear'] - avg_errors[method]) / avg_errors['linear'] * 100
            vs_fx = (avg_errors['fx_average'] - avg_errors[method]) / avg_errors['fx_average'] * 100
            print(f"{name:<15} {avg_errors[method]:<15.4f} {vs_linear:+.1f}%{'':<6} {vs_fx:+.1f}%")

        print('='*70)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Bar chart by method
        ax = axes[0]
        x = np.arange(len(method_names))
        colors = ['gray', 'orange', 'green', 'steelblue', 'darkblue']
        bars = ax.bar(x, [avg_errors[m] for m in methods], color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Error (m)')
        ax.set_title('Prediction Error by Method')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, [avg_errors[m] for m in methods]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=9)

        # Improvement vs Linear
        ax = axes[1]
        improvements = [(avg_errors['linear'] - avg_errors[m]) / avg_errors['linear'] * 100
                       for m in methods]
        colors_imp = ['forestgreen' if imp > 0 else 'crimson' for imp in improvements]
        ax.bar(x, improvements, color=colors_imp, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylabel('Improvement vs Linear (%)')
        ax.set_title('Improvement over Linear Extrapolation')
        ax.grid(True, alpha=0.3, axis='y')

        # Per-pedestrian comparison
        ax = axes[2]
        width = 0.15
        x_ped = np.arange(len(all_results))
        for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
            vals = [r[method] for r in all_results]
            ax.bar(x_ped + i*width - 2*width, vals, width, label=name, color=color, alpha=0.8)
        ax.set_xticks(x_ped)
        ax.set_xticklabels([f"Ped {int(r['ped_id'])}" for r in all_results], rotation=45)
        ax.set_ylabel('Mean Error (m)')
        ax.set_title('Error by Pedestrian')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('ETH Dataset: ESN vs Conventional Methods\n'
                    f'{len(all_results)} Pedestrians, {args.future_horizon}-step Horizon',
                    fontsize=14)
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"eth_method_comparison_{timestamp}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved: {out_path}")


if __name__ == "__main__":
    main()
