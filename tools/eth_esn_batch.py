#!/usr/bin/env python3
"""
ETH Dataset ESN Batch Evaluation
No matplotlib dependency - just computes error metrics
"""
import sys
import os
# Add user site-packages first to use updated scipy
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.10/site-packages'))

import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, RLS


class OnlineStandardizer:
    def __init__(self, mean, var, alpha=0.02):
        self.mean = np.array(mean, dtype=float)
        self.var = np.array(var, dtype=float)
        self.alpha = alpha
        self.eps = 1e-6

    def update(self, x):
        x = np.atleast_2d(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1 - self.alpha) * self.var + self.alpha * batch_var

    def transform(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.eps)

    def inverse_transform(self, xs):
        return xs * np.sqrt(self.var + self.eps) + self.mean


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

        reservoir = Reservoir(
            units=units,
            sr=sr,
            lr=lr,
            input_scaling=input_scaling,
            bias=bias,
            seed=int(seed + i)
        )
        readout = RLS(forgetting=rls_forgetting)
        esn = reservoir >> readout
        esns.append(esn)
        print(f"  ESN {i}: units={units}, sr={sr:.3f}, lr={lr:.3f}")
    return esns


def evaluate_pedestrian(path, ped_id, warmup=5, window=20, n_models=10, future_horizon=20, seed=42):
    """Evaluate ESN on a single pedestrian trajectory."""
    df = load_eth_dataset(path)
    traj = df[df["ped_id"] == ped_id][["x", "y"]].to_numpy()

    if len(traj) < warmup + future_horizon + 10:
        print(f"  Skipping ped {ped_id}: insufficient data ({len(traj)} frames)")
        return None

    # Create ESNs
    esns = create_diverse_esns(n_models=n_models, base_units=25, seed=seed, rls_forgetting=0.99)

    # Warmup
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

    # Adaptation config
    adapt_window = 5
    delta_window = 6
    adapt_damping_nominal = 0.35
    err_clip = 1.0

    esn_err_list = []
    window_buffer = list(warmup_buffer)

    # Process remaining trajectory
    for frame_idx in range(warmup, len(traj) - future_horizon):
        pos = traj[frame_idx]
        window_buffer.append(pos)
        if len(window_buffer) > window:
            window_buffer.pop(0)

        in_online_std.update(pos.reshape(1, -1))

        X_recent = np.array(window_buffer[-adapt_window:])
        X_recent_s = in_online_std.transform(X_recent)

        # ESN predictions
        all_esn_preds = []
        for i, esn in enumerate(esns):
            last_input = X_recent_s.copy()
            future_preds = []
            for _ in range(future_horizon):
                try:
                    delta_s = esn.run(last_input)[-1]
                except Exception:
                    delta_s = np.zeros((1, 2))
                delta = tg_online_std.inverse_transform(delta_s.reshape(1, -1))[0]
                last_pos = in_online_std.inverse_transform(last_input[-1].reshape(1, -1))[0]
                next_pos_abs = last_pos + delta
                future_preds.append(next_pos_abs)
                last_input = np.roll(last_input, -1, axis=0)
                last_input[-1] = in_online_std.transform(next_pos_abs.reshape(1, -1))[0]
            all_esn_preds.append(future_preds)

            # Adaptation
            if len(X_recent_s) > 1:
                fit_len = min(delta_window, len(X_recent_s) - 1)
                adapt_X = X_recent_s[-fit_len-1:-1]
                adapt_Y = np.diff(X_recent_s[-fit_len-1:], axis=0)
                adapt_Y_adj = np.clip(adapt_Y * adapt_damping_nominal, -err_clip, err_clip)
                try:
                    esn.partial_fit(adapt_X, adapt_Y_adj)
                except Exception:
                    pass

        if len(window_buffer) >= 2:
            last_real_delta = np.array(window_buffer[-1]) - np.array(window_buffer[-2])
            tg_online_std.update(last_real_delta.reshape(1, -1))

        # Compute ESN average
        esn_avg = np.mean(np.array(all_esn_preds), axis=0)

        # Ground truth
        gt_future = traj[frame_idx + 1:frame_idx + 1 + future_horizon]

        if len(gt_future) >= future_horizon:
            err_esn = np.mean(np.linalg.norm(esn_avg - gt_future, axis=1))
            if np.isfinite(err_esn):
                esn_err_list.append(err_esn)

    if len(esn_err_list) > 0:
        return {
            'ped_id': ped_id,
            'mean_error': np.mean(esn_err_list),
            'std_error': np.std(esn_err_list),
            'n_frames': len(esn_err_list),
            'traj_length': len(traj)
        }
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ETH Pedestrian ESN Batch Evaluation')
    parser.add_argument('--data', default='data/students001_train.txt', help='Path to dataset')
    parser.add_argument('--ped_ids', type=int, nargs='+', default=None, help='Pedestrian IDs (default: auto-select)')
    parser.add_argument('--n_peds', type=int, default=5, help='Number of pedestrians to evaluate')
    parser.add_argument('--n_models', type=int, default=10, help='Number of ESN models')
    parser.add_argument('--future_horizon', type=int, default=20, help='Future prediction steps')
    args = parser.parse_args()

    # Find data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data)
    if not os.path.exists(data_path):
        data_path = args.data

    print("="*60)
    print("ETH Dataset ESN Batch Evaluation")
    print("="*60)
    print(f"Data: {data_path}")
    print(f"ESN models: {args.n_models}")
    print(f"Future horizon: {args.future_horizon} steps")

    df = load_eth_dataset(data_path)

    # Select pedestrians
    if args.ped_ids:
        ped_ids = args.ped_ids
    else:
        # Auto-select pedestrians with sufficient trajectory length
        ped_counts = df.groupby('ped_id').size()
        valid_peds = ped_counts[ped_counts > 50].index.tolist()
        ped_ids = valid_peds[:args.n_peds]

    print(f"Testing pedestrians: {ped_ids}")
    print("="*60)

    all_results = []
    for pid in ped_ids:
        print(f"\nPedestrian {pid}:")
        result = evaluate_pedestrian(
            data_path,
            ped_id=pid,
            warmup=5,
            window=20,
            n_models=args.n_models,
            future_horizon=args.future_horizon
        )
        if result:
            all_results.append(result)
            print(f"  Mean Error: {result['mean_error']:.4f} ± {result['std_error']:.4f} m")
            print(f"  Frames: {result['n_frames']}, Trajectory: {result['traj_length']}")

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"{'Ped ID':<10} {'Mean Error (m)':<15} {'Std':<10} {'Frames':<10}")
        print('-'*45)
        total_err = []
        for r in all_results:
            print(f"{r['ped_id']:<10} {r['mean_error']:<15.4f} {r['std_error']:<10.4f} {r['n_frames']:<10}")
            total_err.append(r['mean_error'])
        print('-'*45)
        print(f"{'Average':<10} {np.mean(total_err):<15.4f} {np.std(total_err):<10.4f}")
        print('='*60)


if __name__ == "__main__":
    main()
