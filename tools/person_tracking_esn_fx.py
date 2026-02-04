#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.10/site-packages'))

import matplotlib
matplotlib.use('Agg')  # Headless backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, RLS

from scipy.optimize import curve_fit
from collections import deque

class OnlineStandardizer:
    def __init__(self, mean, var, alpha=0.02):
        self.mean = np.array(mean, dtype=float)
        self.var = np.array(var, dtype=float)
        self.alpha = alpha
        self.eps = 1e-6

    def update(self, x):
        # x can be 1D or (N, D)
        x = np.atleast_2d(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1 - self.alpha) * self.var + self.alpha * batch_var

    def transform(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.eps)

    def inverse_transform(self, xs):
        return xs * np.sqrt(self.var + self.eps) + self.mean

# ----------------------------
# ETH Dataset Loader/Streamer
# ----------------------------
def load_eth_dataset(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["frame", "ped_id", "x", "y"]
    return df

def stream_eth_positions(df, ped_id=1):
    df_ped = df[df["ped_id"] == ped_id].sort_values("frame")
    for _, row in df_ped.iterrows():
        yield np.array([row["x"], row["y"]])

# ----------------------------
# Create multiple ESNs.
# ----------------------------
def create_diverse_esns(n_models=5, base_units=25, seed=42, rls_forgetting=0.95):
    """
    Drop-in ESN creation: stable ranges but slightly shorter memory (lower sr, higher leak).
    """
    esns = []
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        units = base_units + int(rng.integers(-5, 6))
        sr = float(rng.uniform(0.8, 0.9))        # more conservative spectral radius
        lr = float(rng.uniform(0.35, 0.6))       # leak a bit larger -> less long memory
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
        print(f"ESN {i}: units={units}, sr={sr:.3f}, lr={lr:.3f}, inscale={input_scaling:.3f}, rls_f={rls_forgetting}")
    return esns

# ----------------------------
# Batch evaluation (no animation)
# ----------------------------
def run_batch_evaluation(path, ped_id=1, warmup=5, window=10,
                         n_models=3, future_horizon=10, seed=42):
    """Run ESN evaluation without animation, return error metrics."""
    df = load_eth_dataset(path)
    traj = df[df["ped_id"] == ped_id][["x", "y"]].to_numpy()

    if len(traj) < warmup + future_horizon:
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

    err_deques = [deque(maxlen=8) for _ in esns]
    esn_err_list = []
    fx_err_list = []

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
            'n_frames': len(esn_err_list)
        }
    return None

# ----------------------------
# Online ESN with Ridge
# ----------------------------
def run_online_esn_multi(path, ped_id=1, warmup=5, window=10,
                         n_models=3, future_horizon=10, seed=42,
                         save_mp4=False, filename=None, fps=5):
    if filename is None:
        filename = f"eth_pedestrian_{ped_id}_prediction.mp4"
 
    df = load_eth_dataset(path)
    stream = stream_eth_positions(df, ped_id)
    traj = df[df["ped_id"] == ped_id][["x", "y"]].to_numpy()

    # --- Create ESNs
    esns = create_diverse_esns(n_models=n_models, base_units=25, seed=seed, rls_forgetting=0.99)

    history = []

    # --- Warmup buffer
    warmup_buffer = []
    for _ in range(warmup):
        try:
            pos = next(stream)
        except StopIteration:
            break
        warmup_buffer.append(pos)
        history.append(pos)
    warmup_buffer = np.array(warmup_buffer)
    if len(warmup_buffer) < 2:
        raise RuntimeError("Not enough warmup data for online ESN initialization")

    # --- Preprocessing (fit on warmup)
    X_warm = warmup_buffer[:-1]
    Y_warm = warmup_buffer[1:] - warmup_buffer[:-1]
    scaler_in = StandardScaler().fit(X_warm)
    scaler_tg = StandardScaler().fit(Y_warm)

    # --- Online-friendly standardizers (EWMA initialized from batch scalers)
    in_online_std = OnlineStandardizer(scaler_in.mean_, scaler_in.var_, alpha=0.02)
    tg_online_std = OnlineStandardizer(scaler_tg.mean_, scaler_tg.var_, alpha=0.02)

    X_w_s = in_online_std.transform(X_warm)
    Y_w_s = tg_online_std.transform(Y_warm)

    # --- Warm-start partial_fit
    for esn in esns:
        esn.partial_fit(X_w_s, Y_w_s)

    # --- Plot setup ---
    fig, ax = plt.subplots()
    ax.set_xlim(df["x"].min() - 5, df["x"].max() + 5)
    ax.set_ylim(df["y"].min() - 5, df["y"].max() + 5)
    ax.set_title(f"ETH Pedestrian {ped_id}: Online ESN Prediction")
    real_dot, = ax.plot([], [], "bo", label="Current")
    history_line, = ax.plot([], [], "g-", label="History")
    pred_lines = [ax.plot([], [], "--", label=f"ESN {i+1}", alpha=0.4)[0] for i in range(n_models)]
    fx_line, = ax.plot([], [], "r--", linewidth=2, label="f(x) avg (3 fits)")
    esn_avg_line, = ax.plot([], [], "m-", linewidth=2, label="Avg ESN")
    ax.legend()
    window_buffer = list(warmup_buffer)

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

    # --- helper functions for f(x) ---
    def sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def fit_sigmoid(x, y):
        from scipy.optimize import curve_fit
        p0 = [max(y)-min(y), np.median(x), 1, min(y)]
        try:
            popt, _ = curve_fit(sigmoid, x, y, p0, maxfev=5000)
            return popt
        except:
            return [max(y)-min(y), np.median(x), 1, min(y)]

    esn_err_list = []
    fx_err_list = []

    def update(frame_idx):
        nonlocal window_buffer, history

        try:
            pos = next(stream)
        except StopIteration:
            return [real_dot, history_line, fx_line, esn_avg_line] + pred_lines

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

        # --- f(x) predictor (3 fits + average) ---
        if len(window_buffer) >= window:
            hist_arr = np.array(window_buffer[-window:])
            xs = np.arange(len(hist_arr))
            ys_x = hist_arr[:, 0]
            ys_y = hist_arr[:, 1]

            # linear and parabolic fits
            lin_x = np.polyfit(xs, ys_x, 1)
            lin_y = np.polyfit(xs, ys_y, 1)
            par_x = np.polyfit(xs, ys_x, 2)
            par_y = np.polyfit(xs, ys_y, 2)

            # sigmoid fits
            sig_popt_x = fit_sigmoid(xs, ys_x)
            sig_popt_y = fit_sigmoid(xs, ys_y)

            # extrapolate into future
            x_future = np.arange(len(hist_arr), len(hist_arr) + future_horizon)
            lin_pred_x = np.polyval(lin_x, x_future)
            lin_pred_y = np.polyval(lin_y, x_future)
            par_pred_x = np.polyval(par_x, x_future)
            par_pred_y = np.polyval(par_y, x_future)
            sig_pred_x = sigmoid(x_future, *sig_popt_x)
            sig_pred_y = sigmoid(x_future, *sig_popt_y)

            # average
            fx_pred_x = (lin_pred_x + par_pred_x + sig_pred_x) / 3
            fx_pred_y = (lin_pred_y + par_pred_y + sig_pred_y) / 3
            f_pred_points = np.column_stack((fx_pred_x, fx_pred_y))
        else:
            f_pred_points = np.empty((0, 2))

        # --- ESN predictions ---
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

                # roll forward
                last_input = np.roll(last_input, -1, axis=0)
                last_input[-1] = in_online_std.transform(next_pos_abs.reshape(1, -1))[0]

            all_esn_preds.append(future_preds)

            # plot individual ESN predictions
            pred_lines[i].set_data(
                [pos[0]] + [p[0] for p in future_preds],
                [pos[1]] + [p[1] for p in future_preds]
            )

            # --- adaptation logic ---
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
                except Exception:
                    pass
                
                # clip readout weights and reservoir state (defensive checks)
                if hasattr(esn, "W_out"):
                    esn.W_out = np.clip(esn.W_out, -wout_clip, wout_clip)
                if hasattr(esn, "res") and hasattr(esn.res, "state"):
                    esn.res.state = np.clip(esn.res.state, -state_clip, state_clip)
                if len(window_buffer) >= 2:
                    last_real_delta = np.array(window_buffer[-1]) - np.array(window_buffer[-2])
                    tg_online_std.update(last_real_delta.reshape(1, -1))

        # --- compute average ESN prediction ---
        if len(all_esn_preds) > 0:
            esn_avg = np.mean(np.array(all_esn_preds), axis=0)
        else:
            esn_avg = np.empty((0, 2))

        # --- error computation (using known ground truth) ---
        gt_future_start = frame_idx + 1
        gt_future_end = frame_idx + 1 + future_horizon
        gt_future = traj[gt_future_start:gt_future_end, :2]
        if len(gt_future) > 1:
            L = min(len(gt_future), len(esn_avg), len(f_pred_points))
            gt = gt_future[:L]
            err_esn_vs_gt = np.mean(np.linalg.norm(esn_avg[:L] - gt, axis=1))
            err_fx_vs_gt = np.mean(np.linalg.norm(f_pred_points[:L] - gt, axis=1))

            # --- Store frame errors ---
            if np.isfinite(err_esn_vs_gt):
                esn_err_list.append(err_esn_vs_gt)
            if np.isfinite(err_fx_vs_gt):
                fx_err_list.append(err_fx_vs_gt)

            #if not save_mp4:
            #    print(f"Frame {frame_idx:03d} | Err(ESN,GT)={err_esn_vs_gt:.3f} | Err(f(x),GT)={err_fx_vs_gt:.3f}")

        # --- update plots ---
        fx_line.set_data(f_pred_points[:, 0], f_pred_points[:, 1])
        esn_avg_line.set_data(esn_avg[:, 0], esn_avg[:, 1])
        real_dot.set_data(pos[0], pos[1])
        history_line.set_data([p[0] for p in history], [p[1] for p in history])

        return [real_dot, history_line, fx_line, esn_avg_line] + pred_lines

    length = len(df[df["ped_id"] == ped_id])
    ani = FuncAnimation(fig, update, interval=50, blit=True, frames=length)

    if save_mp4:
        ani.save(filename, writer="ffmpeg", fps=fps)
        print(f"Animation saved to {filename}")
    else:
        plt.show()

    if len(esn_err_list) > 0:
        mean_esn_err = np.mean(esn_err_list)
        std_esn_err = np.std(esn_err_list)

        mean_fx_err = np.mean(fx_err_list)
        std_fx_err = np.std(fx_err_list)

        print(f"\n=== FINAL AVERAGE ERRORS ===")
        print(f"Average Err(ESN, GT): {mean_esn_err:.4f} ± {std_esn_err:.4f}")
        print(f"Average Err(f(x), GT): {mean_fx_err:.4f} ± {std_fx_err:.4f}")
        print(f"\n======\n======\n\n\n")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='ETH Pedestrian ESN Path Prediction')
    parser.add_argument('--data', default='./data/students001_train.txt', help='Path to dataset')
    parser.add_argument('--output', default='../output', help='Output directory')
    parser.add_argument('--ped_ids', type=int, nargs='+', default=[399, 168, 269], help='Pedestrian IDs to test')
    parser.add_argument('--n_models', type=int, default=10, help='Number of ESN models')
    parser.add_argument('--future_horizon', type=int, default=20, help='Future prediction steps')
    parser.add_argument('--save_mp4', action='store_true', help='Save as MP4 animation')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_results = []
    for pid in args.ped_ids:
        print(f"\n{'='*60}")
        print(f"Testing Pedestrian ID: {pid}")
        print('='*60)

        if args.save_mp4:
            filename = os.path.join(args.output, f"eth_pedestrian_{pid}.mp4")
            run_online_esn_multi(
                args.data,
                ped_id=pid,
                warmup=5,
                window=20,
                n_models=args.n_models,
                future_horizon=args.future_horizon,
                save_mp4=True,
                filename=filename,
                fps=10
            )
        else:
            # Batch evaluation (no animation)
            result = run_batch_evaluation(
                args.data,
                ped_id=pid,
                warmup=5,
                window=20,
                n_models=args.n_models,
                future_horizon=args.future_horizon
            )
            if result:
                all_results.append(result)
                print(f"  Mean Error: {result['mean_error']:.4f} ± {result['std_error']:.4f} m")
                print(f"  Frames evaluated: {result['n_frames']}")

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY - ETH Dataset ESN Evaluation")
        print('='*60)
        print(f"{'Ped ID':<10} {'Mean Error':<15} {'Std':<10} {'Frames':<10}")
        print('-'*45)
        total_err = []
        for r in all_results:
            print(f"{r['ped_id']:<10} {r['mean_error']:<15.4f} {r['std_error']:<10.4f} {r['n_frames']:<10}")
            total_err.append(r['mean_error'])
        print('-'*45)
        print(f"{'Average':<10} {np.mean(total_err):<15.4f}")
        print('='*60)
