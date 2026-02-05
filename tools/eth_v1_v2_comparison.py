#!/usr/bin/env python3
"""
ETH Dataset: V1 vs V2 (Kalman Hybrid) Comparison
Compare original ESN with Kalman filter hybrid on real pedestrian data
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
    """Simple Kalman Filter for 2D position tracking."""
    def __init__(self, dt=0.1, process_noise=0.1, measurement_noise=0.05):
        self.dt = dt
        # State: [x, y, vx, vy]
        self.x = np.zeros(4)
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        # Process noise
        self.Q = np.eye(4) * process_noise
        # Measurement noise
        self.R = np.eye(2) * measurement_noise
        # Covariance
        self.P = np.eye(4)
        self.initialized = False

    def update(self, z):
        z = np.array(z).flatten()
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            return self.x[:2]

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
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

# ----------------------------
# ETH Dataset Loader/Streamer
# ----------------------------
def load_eth_dataset(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["frame", "ped_id", "x", "y"]
    return df

# ----------------------------
# Create multiple ESNs.
# ----------------------------
def create_diverse_esns(n_models=5, base_units=25, seed=42, rls_forgetting=0.99):
    """
    Drop-in ESN creation: stable ranges but slightly shorter memory (lower sr, higher leak).
    """
    esns = []
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        units = base_units + int(rng.integers(-5, 6))
        sr = float(rng.uniform(0.8, 0.9))
        lr = float(rng.uniform(0.35, 0.6))
        input_scaling = float(rng.uniform(0.2, 0.4))
        bias = float(rng.uniform(-0.2, 0.2))

        reservoir = Reservoir(
            units=units, sr=sr, lr=lr,
            input_scaling=input_scaling, bias=bias,
            seed=int(seed + i)
        )
        readout = RLS(forgetting=rls_forgetting)
        esn = reservoir >> readout
        esns.append(esn)
    return esns

# ----------------------------
# Online ESN with Ridge
# ----------------------------
def evaluate_v1(traj, warmup=5, window=20, n_models=10, future_horizon=20, seed=42):
    """V1: Original ESN only."""
    if len(traj) < warmup + future_horizon + 10:
        return None

    esns = create_diverse_esns(n_models=n_models, base_units=25, seed=seed)

    history = traj[:warmup].tolist()
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

    # --- Adaptation config ---
    adapt_window = 5
    delta_window = 6
    sudden_change_thresh = 0.6
    adapt_damping_nominal = 0.35
    adapt_damping_boost = 1.0
    boost_frames = 6
    boost_error_thresh = 0.5
    state_clip = 5.0
    wout_clip = 8.0
    err_clip = 1.0

    # per-ESN running error buffers and boost counters
    err_deques = [deque(maxlen=8) for _ in esns]
    boost_counters = [0 for _ in esns]

    errors = []
    window_buffer = list(warmup_buffer)

    for frame_idx in range(warmup, len(traj) - future_horizon):
        pos = traj[frame_idx]

        # update history and window
        history.append(pos)
        window_buffer.append(pos)
        if len(window_buffer) > window:
            window_buffer.pop(0)

        # update online scalers
        in_online_std.update(pos.reshape(1, -1))
        
        # detect sudden jumps
        if len(history) > 1:
            last_delta = np.linalg.norm(pos - history[-2])
            if last_delta > sudden_change_thresh:
                for esn in esns:
                    if hasattr(esn, "res"):
                        esn.res.reset_state()

        # Prepare recent (standardized) input window
        X_recent = np.array(window_buffer[-adapt_window:])
        X_recent_s = in_online_std.transform(X_recent)

        all_preds = []
        for i, esn in enumerate(esns):
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

                # roll forward
                last_input = np.roll(last_input, -1, axis=0)
                last_input[-1] = in_online_std.transform(next_pos.reshape(1, -1))[0]

            all_preds.append(future_preds)

            # Adaptation
            if len(X_recent_s) > 1:
                fit_len = min(delta_window, len(X_recent_s) - 1)
                adapt_X = X_recent_s[-fit_len-1:-1]
                adapt_Y = np.diff(X_recent_s[-fit_len-1:], axis=0)

                # predict the last standardized delta over this short window
                try:
                    pred_last_s = esn.run(adapt_X)[-1]
                except Exception:
                    pred_last_s = np.zeros((1, adapt_Y.shape[1]))

                real_last = adapt_Y[-1].reshape(1, -1)
                instantaneous_err = np.linalg.norm(real_last - pred_last_s)

                # update running error + decide damping (boost vs nominal)
                err_deques[i].append(instantaneous_err)
                running_err = np.mean(err_deques[i]) if len(err_deques[i]) > 0 else 0.0


                if boost_counters[i] > 0:
                    damping = adapt_damping_boost
                    boost_counters[i] -= 1
                elif running_err > boost_error_thresh:
                    damping = adapt_damping_boost
                    boost_counters[i] = boost_frames - 1
                else:
                    damping = adapt_damping_nominal

                # damp + clip the target deltas before partial_fit
                adapt_Y_adj = np.clip(adapt_Y * damping, -err_clip, err_clip)
                try:
                    esn.partial_fit(adapt_X, adapt_Y_adj)
                except:
                    pass

                # clip readout weights and reservoir state (defensive checks)
                if hasattr(esn, "W_out"):
                    esn.W_out = np.clip(esn.W_out, -wout_clip, wout_clip)
                if hasattr(esn, "res") and hasattr(esn.res, "state"):
                    esn.res.state = np.clip(esn.res.state, -state_clip, state_clip)
                if len(window_buffer) >= 2:
                    last_delta = np.array(window_buffer[-1]) - np.array(window_buffer[-2])
                    tg_online_std.update(last_delta.reshape(1, -1))

        # --- compute average ESN prediction ---
        if len(all_preds) > 0:
            esn_avg = np.mean(np.array(all_preds), axis=0)
        else:
            esn_avg = np.empty((0, 2))

        # --- error computation (using known ground truth) ---
        gt_future = traj[frame_idx + 1:frame_idx + 1 + future_horizon]
        if len(gt_future) >= future_horizon:
            err = np.mean(np.linalg.norm(esn_avg - gt_future, axis=1))
            if np.isfinite(err):
                errors.append(err)

    return np.array(errors)


def evaluate_v2_kalman(traj, warmup=5, window=20, n_models=10, future_horizon=20,
                       seed=42, kalman_weight=0.3):
    """V2: ESN + Kalman Filter Hybrid."""
    if len(traj) < warmup + future_horizon + 10:
        return None

    esns = create_diverse_esns(n_models=n_models, base_units=25, seed=seed)
    kalman = SimpleKalmanFilter(dt=0.1, process_noise=0.1, measurement_noise=0.05)

    history = traj[:warmup].tolist()
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

    # Initialize Kalman with warmup
    for pos in warmup_buffer:
        kalman.update(pos)

    # --- Adaptation config ---
    adapt_window = 5
    delta_window = 6
    sudden_change_thresh = 0.6
    adapt_damping_nominal = 0.35
    adapt_damping_boost = 1.0
    boost_frames = 6
    boost_error_thresh = 0.5
    state_clip = 5.0
    wout_clip = 8.0
    err_clip = 1.0

    # per-ESN running error buffers and boost counters
    err_deques = [deque(maxlen=8) for _ in esns]
    boost_counters = [0 for _ in esns]

    errors = []
    window_buffer = list(warmup_buffer)

    for frame_idx in range(warmup, len(traj) - future_horizon):
        pos = traj[frame_idx]

        # update history and window
        history.append(pos)
        window_buffer.append(pos)
        if len(window_buffer) > window:
            window_buffer.pop(0)

        # Update Kalman
        kalman.update(pos)

        # update online scalers
        in_online_std.update(pos.reshape(1, -1))
        
        # detect sudden jumps
        if len(history) > 1:
            last_delta = np.linalg.norm(pos - history[-2])
            if last_delta > sudden_change_thresh:
                for esn in esns:
                    if hasattr(esn, "res"):
                        esn.res.reset_state()

        # Prepare recent (standardized) input window
        X_recent = np.array(window_buffer[-adapt_window:])
        X_recent_s = in_online_std.transform(X_recent)

        # ESN predictions
        all_preds = []
        for i, esn in enumerate(esns):
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

                # roll forward
                last_input = np.roll(last_input, -1, axis=0)
                last_input[-1] = in_online_std.transform(next_pos.reshape(1, -1))[0]

            all_preds.append(future_preds)

            # Adaptation
            if len(X_recent_s) > 1:
                fit_len = min(delta_window, len(X_recent_s) - 1)
                adapt_X = X_recent_s[-fit_len-1:-1]
                adapt_Y = np.diff(X_recent_s[-fit_len-1:], axis=0)

                # predict the last standardized delta over this short window
                try:
                    pred_last_s = esn.run(adapt_X)[-1]
                except Exception:
                    pred_last_s = np.zeros((1, adapt_Y.shape[1]))

                real_last = adapt_Y[-1].reshape(1, -1)
                instantaneous_err = np.linalg.norm(real_last - pred_last_s)

                # update running error + decide damping (boost vs nominal)
                err_deques[i].append(instantaneous_err)
                running_err = np.mean(err_deques[i]) if len(err_deques[i]) > 0 else 0.0


                if boost_counters[i] > 0:
                    damping = adapt_damping_boost
                    boost_counters[i] -= 1
                elif running_err > boost_error_thresh:
                    damping = adapt_damping_boost
                    boost_counters[i] = boost_frames - 1
                else:
                    damping = adapt_damping_nominal

                # damp + clip the target deltas before partial_fit
                adapt_Y_adj = np.clip(adapt_Y * damping, -err_clip, err_clip)
                try:
                    esn.partial_fit(adapt_X, adapt_Y_adj)
                except:
                    pass

                # clip readout weights and reservoir state (defensive checks)
                if hasattr(esn, "W_out"):
                    esn.W_out = np.clip(esn.W_out, -wout_clip, wout_clip)
                if hasattr(esn, "res") and hasattr(esn.res, "state"):
                    esn.res.state = np.clip(esn.res.state, -state_clip, state_clip)
                if len(window_buffer) >= 2:
                    last_delta = np.array(window_buffer[-1]) - np.array(window_buffer[-2])
                    tg_online_std.update(last_delta.reshape(1, -1))

        esn_avg = np.mean(np.array(all_preds), axis=0)

        # Kalman predictions
        kalman_pred = kalman.predict_future(future_horizon)

        # Hybrid: weighted combination
        hybrid_pred = (1 - kalman_weight) * esn_avg + kalman_weight * kalman_pred

        gt_future = traj[frame_idx + 1:frame_idx + 1 + future_horizon]

        if len(gt_future) >= future_horizon:
            err = np.mean(np.linalg.norm(hybrid_pred - gt_future, axis=1))
            if np.isfinite(err):
                errors.append(err)

    return np.array(errors)


def main():
    parser = argparse.ArgumentParser(description='ETH V1 vs V2 Comparison')
    parser.add_argument('--data', default='data/students001_train.txt', help='Dataset path')
    parser.add_argument('--output', default='../output', help='Output directory')
    parser.add_argument('--ped_ids', type=int, nargs='+', default=[399, 168, 269, 177, 178],
                       help='Pedestrian IDs')
    parser.add_argument('--n_models', type=int, default=10, help='Number of ESN models')
    parser.add_argument('--future_horizon', type=int, default=20, help='Prediction steps')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data)
    output_dir = os.path.join(script_dir, args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("ETH Dataset: V1 vs V2 (Kalman Hybrid) Comparison")
    print("="*60)

    df = load_eth_dataset(data_path)

    results = []
    for pid in args.ped_ids:
        print(f"\nPedestrian {pid}:")
        traj = df[df["ped_id"] == pid][["x", "y"]].to_numpy()

        if len(traj) < 50:
            print(f"  Skipping: insufficient data ({len(traj)} frames)")
            continue

        print(f"  Trajectory: {len(traj)} frames")
        print(f"  Evaluating V1 (ESN only)...")
        v1_errors = evaluate_v1(traj, n_models=args.n_models, future_horizon=args.future_horizon)

        print(f"  Evaluating V2 (Kalman Hybrid)...")
        v2_errors = evaluate_v2_kalman(traj, n_models=args.n_models, future_horizon=args.future_horizon)

        if v1_errors is not None and v2_errors is not None:
            v1_mean = np.mean(v1_errors)
            v2_mean = np.mean(v2_errors)
            improvement = (v1_mean - v2_mean) / v1_mean * 100

            results.append({
                'ped_id': pid,
                'v1_mean': v1_mean,
                'v1_std': np.std(v1_errors),
                'v2_mean': v2_mean,
                'v2_std': np.std(v2_errors),
                'improvement': improvement,
                'n_frames': len(v1_errors)
            })

            print(f"  V1: {v1_mean:.4f}m, V2: {v2_mean:.4f}m, Improvement: {improvement:+.1f}%")

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"{'Ped ID':<10} {'V1 (m)':<12} {'V2 (m)':<12} {'Improvement':<12}")
        print('-'*46)

        v1_all = []
        v2_all = []
        for r in results:
            print(f"{int(r['ped_id']):<10} {r['v1_mean']:<12.4f} {r['v2_mean']:<12.4f} {r['improvement']:+.1f}%")
            v1_all.append(r['v1_mean'])
            v2_all.append(r['v2_mean'])

        print('-'*46)
        avg_v1 = np.mean(v1_all)
        avg_v2 = np.mean(v2_all)
        avg_imp = (avg_v1 - avg_v2) / avg_v1 * 100
        print(f"{'Average':<10} {avg_v1:<12.4f} {avg_v2:<12.4f} {avg_imp:+.1f}%")
        print('='*60)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Bar comparison
        ax = axes[0]
        x = np.arange(len(results))
        width = 0.35
        ax.bar(x - width/2, [r['v1_mean'] for r in results], width, label='V1 (ESN)', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, [r['v2_mean'] for r in results], width, label='V2 (Kalman Hybrid)', color='forestgreen', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Ped {int(r['ped_id'])}" for r in results], rotation=45)
        ax.set_ylabel('Mean Error (m)')
        ax.set_title('V1 vs V2 by Pedestrian')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Improvement chart
        ax = axes[1]
        improvements = [r['improvement'] for r in results]
        colors = ['forestgreen' if imp > 0 else 'crimson' for imp in improvements]
        ax.bar(x, improvements, color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=np.mean(improvements), color='blue', linestyle='--',
                   label=f'Avg: {np.mean(improvements):+.1f}%')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Ped {int(r['ped_id'])}" for r in results], rotation=45)
        ax.set_ylabel('Improvement (%)')
        ax.set_title('V2 Improvement over V1')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Summary
        ax = axes[2]
        ax.bar(['V1\n(ESN)', 'V2\n(Kalman Hybrid)'], [avg_v1, avg_v2],
               color=['steelblue', 'forestgreen'], alpha=0.8)
        ax.set_ylabel('Average Error (m)')
        ax.set_title(f'Overall Comparison\n(Improvement: {avg_imp:+.1f}%)')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate([avg_v1, avg_v2]):
            ax.text(i, v + 0.02, f'{v:.3f}m', ha='center', fontsize=12)

        plt.suptitle('ETH Dataset: V1 vs V2 (Kalman Hybrid) Comparison\n'
                     f'{len(results)} Pedestrians, {args.n_models} ESN Models, {args.future_horizon}-step Horizon',
                     fontsize=14)
        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"eth_v1_v2_comparison_{timestamp}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nVisualization saved: {out_path}")


if __name__ == "__main__":
    main()
