#!/usr/bin/env python3
"""
ETH Dataset ESN Visualization
Visualize ESN predictions on real pedestrian trajectories
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
            units=units, sr=sr, lr=lr,
            input_scaling=input_scaling, bias=bias,
            seed=int(seed + i)
        )
        readout = RLS(forgetting=rls_forgetting)
        esn = reservoir >> readout
        esns.append(esn)
    return esns


def evaluate_and_visualize(path, ped_id, warmup=5, window=20, n_models=10,
                           future_horizon=20, seed=42, visualize_frames=None):
    """Evaluate ESN and collect visualization data."""
    df = load_eth_dataset(path)
    traj = df[df["ped_id"] == ped_id][["x", "y"]].to_numpy()

    if len(traj) < warmup + future_horizon + 10:
        return None

    # Create ESNs
    esns = create_diverse_esns(n_models=n_models, base_units=25, seed=seed)

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

    # Config
    adapt_window = 5
    delta_window = 6
    adapt_damping_nominal = 0.35
    err_clip = 1.0

    esn_err_list = []
    window_buffer = list(warmup_buffer)

    # Store visualization data
    vis_data = []

    # Select frames to visualize
    total_frames = len(traj) - future_horizon - warmup
    if visualize_frames is None:
        # Select 5 evenly spaced frames
        visualize_frames = [int(i * total_frames / 6) for i in range(1, 6)]

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

            # Store visualization data for selected frames
            rel_frame = frame_idx - warmup
            if rel_frame in visualize_frames:
                vis_data.append({
                    'frame': frame_idx,
                    'history': np.array(window_buffer),
                    'current_pos': pos,
                    'prediction': esn_avg,
                    'ground_truth': gt_future,
                    'error': err_esn
                })

    result = {
        'ped_id': ped_id,
        'mean_error': np.mean(esn_err_list) if esn_err_list else None,
        'std_error': np.std(esn_err_list) if esn_err_list else None,
        'n_frames': len(esn_err_list),
        'traj_length': len(traj),
        'full_trajectory': traj,
        'errors': esn_err_list,
        'vis_data': vis_data
    }
    return result


def plot_single_pedestrian(result, output_path):
    """Plot trajectory and predictions for a single pedestrian."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ped_id = result['ped_id']
    traj = result['full_trajectory']
    vis_data = result['vis_data']

    # Plot full trajectory
    ax = axes[0, 0]
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.5, label='Full trajectory')
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
    ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Pedestrian {int(ped_id)} - Full Trajectory')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot predictions at different time points
    for i, vd in enumerate(vis_data[:5]):
        ax = axes[(i+1)//3, (i+1)%3]

        # History
        hist = vd['history']
        ax.plot(hist[:, 0], hist[:, 1], 'b-', linewidth=2, label='History')

        # Current position
        ax.plot(vd['current_pos'][0], vd['current_pos'][1], 'ko', markersize=10, label='Current')

        # Ground truth future
        gt = vd['ground_truth']
        ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2, label='Ground truth')
        ax.plot(gt[-1, 0], gt[-1, 1], 'g^', markersize=8)

        # Prediction
        pred = vd['prediction']
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='ESN prediction')
        ax.plot(pred[-1, 0], pred[-1, 1], 'r^', markersize=8)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Frame {vd["frame"]} (Error: {vd["error"]:.2f}m)')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'ESN Path Prediction - Pedestrian {int(ped_id)}\n'
                 f'Mean Error: {result["mean_error"]:.3f}m, Frames: {result["n_frames"]}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def plot_summary(all_results, output_path):
    """Plot summary of all pedestrians."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Error bar chart
    ax = axes[0, 0]
    ped_ids = [r['ped_id'] for r in all_results]
    mean_errors = [r['mean_error'] for r in all_results]
    std_errors = [r['std_error'] for r in all_results]

    x = np.arange(len(ped_ids))
    bars = ax.bar(x, mean_errors, yerr=std_errors, capsize=5, color='steelblue', alpha=0.7)
    ax.axhline(y=np.mean(mean_errors), color='r', linestyle='--', label=f'Average: {np.mean(mean_errors):.3f}m')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ped {int(p)}' for p in ped_ids], rotation=45)
    ax.set_ylabel('Prediction Error (m)')
    ax.set_title('Mean Prediction Error by Pedestrian')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Trajectory lengths
    ax = axes[0, 1]
    traj_lengths = [r['traj_length'] for r in all_results]
    ax.bar(x, traj_lengths, color='forestgreen', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ped {int(p)}' for p in ped_ids], rotation=45)
    ax.set_ylabel('Trajectory Length (frames)')
    ax.set_title('Trajectory Length by Pedestrian')
    ax.grid(True, alpha=0.3, axis='y')

    # Error vs trajectory length scatter
    ax = axes[1, 0]
    ax.scatter(traj_lengths, mean_errors, s=100, c='steelblue', alpha=0.7)
    for i, pid in enumerate(ped_ids):
        ax.annotate(f'{int(pid)}', (traj_lengths[i], mean_errors[i]),
                   textcoords="offset points", xytext=(5,5), fontsize=9)
    ax.set_xlabel('Trajectory Length (frames)')
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('Error vs Trajectory Length')
    ax.grid(True, alpha=0.3)

    # Error distribution (combined)
    ax = axes[1, 1]
    all_errors = []
    for r in all_results:
        all_errors.extend(r['errors'])
    ax.hist(all_errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(all_errors), color='r', linestyle='--',
               label=f'Mean: {np.mean(all_errors):.3f}m')
    ax.axvline(x=np.median(all_errors), color='g', linestyle='--',
               label=f'Median: {np.median(all_errors):.3f}m')
    ax.set_xlabel('Prediction Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution (All Pedestrians)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('ETH Dataset ESN Evaluation Summary\n'
                 f'{len(all_results)} Pedestrians, 10 ESN Models, 20-step Horizon',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description='ETH Pedestrian ESN Visualization')
    parser.add_argument('--data', default='data/students001_train.txt', help='Path to dataset')
    parser.add_argument('--output', default='../output', help='Output directory')
    parser.add_argument('--ped_ids', type=int, nargs='+', default=[399, 168, 269, 177, 178],
                       help='Pedestrian IDs to visualize')
    parser.add_argument('--n_models', type=int, default=10, help='Number of ESN models')
    parser.add_argument('--future_horizon', type=int, default=20, help='Future prediction steps')
    args = parser.parse_args()

    # Find paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data)
    output_dir = os.path.join(script_dir, args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("ETH Dataset ESN Visualization")
    print("="*60)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Pedestrians: {args.ped_ids}")

    all_results = []
    for pid in args.ped_ids:
        print(f"\nProcessing Pedestrian {pid}...")
        result = evaluate_and_visualize(
            data_path,
            ped_id=pid,
            warmup=5,
            window=20,
            n_models=args.n_models,
            future_horizon=args.future_horizon
        )
        if result and result['mean_error'] is not None:
            all_results.append(result)
            print(f"  Mean Error: {result['mean_error']:.4f}m")

            # Plot individual pedestrian
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(output_dir, f"eth_ped_{int(pid)}_{timestamp}.png")
            plot_single_pedestrian(result, out_path)
            print(f"  Saved: {out_path}")

    # Plot summary
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(output_dir, f"eth_summary_{timestamp}.png")
        plot_summary(all_results, summary_path)
        print(f"\nSummary saved: {summary_path}")

        # Print summary table
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"{'Ped ID':<10} {'Mean Error (m)':<15} {'Std':<10} {'Frames':<10}")
        print('-'*45)
        for r in all_results:
            print(f"{int(r['ped_id']):<10} {r['mean_error']:<15.4f} {r['std_error']:<10.4f} {r['n_frames']:<10}")
        print('-'*45)
        avg_err = np.mean([r['mean_error'] for r in all_results])
        print(f"{'Average':<10} {avg_err:<15.4f}")
        print('='*60)


if __name__ == "__main__":
    main()
