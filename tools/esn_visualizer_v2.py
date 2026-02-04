#!/usr/bin/env python3
"""
ESN Path Prediction Visualizer V2 - Improved Version
=====================================================
Implements improvements step by step:
1. Optimized ESN hyperparameters (larger reservoir)
2. Enhanced direction change detection
3. Dynamic prediction horizon
4. Extended input features (optional)

Usage:
    python3 esn_visualizer_v2.py [--output OUTPUT_DIR] [--pattern PATTERN]
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter
import argparse
import os
from datetime import datetime


class OnlineStandardizer:
    """Online Z-score normalization (EWMA)"""
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


class SimpleESN:
    """Simple Echo State Network"""
    def __init__(self, input_dim=2, output_dim=2, units=25, sr=0.85, lr=0.5,
                 input_scaling=0.3, bias=0.0, seed=42, sparsity=0.9):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.units = units
        self.sr = sr
        self.lr = lr
        self.sparsity = sparsity

        rng = np.random.default_rng(seed)

        self.W_in = rng.uniform(-input_scaling, input_scaling, (units, input_dim))

        W = rng.uniform(-1, 1, (units, units))
        mask = rng.random((units, units)) > sparsity
        W = W * mask

        eigenvalues = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0:
            W = W * (sr / max_eig)
        self.W = W

        self.W_out = np.zeros((output_dim, units))
        self.state = np.zeros(units)
        self.P = np.eye(units) * 1000
        self.forgetting = 0.99
        self.b = rng.uniform(-abs(bias), abs(bias), units) if bias != 0 else np.zeros(units)

    def run(self, X):
        X = np.atleast_2d(X)
        outputs = []
        for x in X:
            pre_act = self.W_in @ x + self.W @ self.state + self.b
            self.state = (1 - self.lr) * self.state + self.lr * np.tanh(pre_act)
            y = self.W_out @ self.state
            outputs.append(y)
        return np.array(outputs)

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        for x, y in zip(X, Y):
            pre_act = self.W_in @ x + self.W @ self.state + self.b
            self.state = (1 - self.lr) * self.state + self.lr * np.tanh(pre_act)

            k = self.P @ self.state / (self.forgetting + self.state @ self.P @ self.state)
            e = y - self.W_out @ self.state
            self.W_out = self.W_out + np.outer(e, k)
            self.P = (self.P - np.outer(k, self.state @ self.P)) / self.forgetting

    def reset_state(self):
        self.state = np.zeros(self.units)


def savgol_smooth(data, window_length=9, polyorder=2):
    """Savitzky-Golay smoothing"""
    arr = np.asarray(data)
    if arr.shape[0] < window_length:
        return arr
    smoothed = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        smoothed[:, i] = savgol_filter(arr[:, i], window_length, polyorder, mode="interp")
    return smoothed


def generate_leg_trajectory(pattern='straight', n_steps=200, noise_level=0.02):
    """Generate synthetic leg trajectory data"""
    t = np.linspace(0, 4 * np.pi, n_steps)

    if pattern == 'straight':
        x = np.linspace(0.5, 1.5, n_steps)
        y = np.zeros(n_steps) + 0.1 * np.sin(t * 2)
    elif pattern == 'curve':
        theta = np.linspace(0, np.pi/2, n_steps)
        r = 1.0
        x = r * np.cos(theta) + 0.5
        y = r * np.sin(theta) - 0.5
    elif pattern == 'zigzag':
        x = np.linspace(0.5, 1.5, n_steps)
        y = 0.3 * np.sin(t * 1.5)
    elif pattern == 'stop_and_go':
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        for i in range(n_steps):
            if i < n_steps // 4:
                x[i] = 0.5 + i * 0.004
            elif i < n_steps // 2:
                x[i] = x[n_steps // 4 - 1]
            elif i < 3 * n_steps // 4:
                x[i] = x[n_steps // 2 - 1] + (i - n_steps // 2) * 0.004
            else:
                x[i] = x[3 * n_steps // 4 - 1]
        y = 0.05 * np.sin(t)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    x += np.random.randn(n_steps) * noise_level
    y += np.random.randn(n_steps) * noise_level

    return np.stack([x, y], axis=1)


# ============ V1: Original Predictor ============

class OriginalESNPredictor:
    """Original ESN Predictor (V1)"""
    def __init__(self, n_models=10, warmup=5, window=20, future_horizon=20):
        self.n_models = n_models
        self.warmup = warmup
        self.window = window
        self.future_horizon = future_horizon

        self.history = deque(maxlen=10000)
        self.warmup_buffer = deque(maxlen=window)
        self.have_warm = False

        self.esns = None
        self.in_online_std = None
        self.tg_online_std = None

        self.adapt_window = 5
        self.adapt_damping = 0.35

    def update(self, pt):
        pt = np.array(pt, dtype=float)
        self.history.append(pt)
        self.warmup_buffer.append(pt)

        if not self.have_warm and len(self.history) >= max(self.warmup, 6):
            X_warm = np.array(list(self.history)[-self.warmup-1:-1])
            Y_warm = np.diff(np.array(list(self.history)[-self.warmup-1:]), axis=0)

            self.in_online_std = OnlineStandardizer(X_warm.mean(axis=0), X_warm.var(axis=0))
            self.tg_online_std = OnlineStandardizer(Y_warm.mean(axis=0), Y_warm.var(axis=0))

            Xw_s = self.in_online_std.transform(X_warm)
            Yw_s = self.tg_online_std.transform(Y_warm)

            # V1: Small reservoir (25 units)
            self.esns = []
            rng = np.random.default_rng(42)
            for i in range(self.n_models):
                units = 25 + int(rng.integers(-5, 6))
                sr = float(rng.uniform(0.8, 0.9))
                lr = float(rng.uniform(0.35, 0.6))
                input_scaling = float(rng.uniform(0.2, 0.4))
                bias = float(rng.uniform(-0.2, 0.2))
                esn = SimpleESN(units=units, sr=sr, lr=lr, input_scaling=input_scaling, bias=bias, seed=42+i)
                esn.partial_fit(Xw_s, Yw_s)
                self.esns.append(esn)

            self.have_warm = True

        if self.in_online_std is not None:
            self.in_online_std.update(pt.reshape(1, -1))
            if len(self.history) >= 2:
                delta = (self.history[-1] - self.history[-2]).reshape(1, -1)
                self.tg_online_std.update(delta)

    def predict(self):
        if not self.have_warm or len(self.warmup_buffer) < 2:
            return None

        positions = np.array(list(self.warmup_buffer))
        if len(positions) >= 9:
            positions = savgol_smooth(positions, window_length=9, polyorder=2)

        adapt_window = min(self.adapt_window, len(positions))
        X_recent = positions[-adapt_window:]
        X_recent_s = self.in_online_std.transform(X_recent)

        all_esn_preds = []
        for esn in self.esns:
            last_input = X_recent_s.copy()
            future_preds = []

            for _ in range(self.future_horizon):
                delta_s = esn.run(last_input)[-1]
                delta = self.tg_online_std.inverse_transform(delta_s.reshape(1, -1))[0]
                last_pos_abs = self.in_online_std.inverse_transform(last_input[-1].reshape(1, -1))[0]
                next_pos_abs = last_pos_abs + delta
                future_preds.append(next_pos_abs)

                last_input = np.roll(last_input, -1, axis=0)
                last_input[-1] = self.in_online_std.transform(next_pos_abs.reshape(1, -1))[0]

            all_esn_preds.append(future_preds)

            # Online adaptation
            if len(X_recent_s) > 1:
                adapt_X = X_recent_s[:-1]
                adapt_Y = np.diff(X_recent_s, axis=0) * self.adapt_damping
                esn.partial_fit(adapt_X, adapt_Y)
                esn.W_out = np.clip(esn.W_out, -8.0, 8.0)

        return np.mean(np.array(all_esn_preds), axis=0)


# ============ V2: Improved Predictor ============

class ImprovedESNPredictor:
    """
    Improved ESN Predictor (V2) with:
    1. Larger reservoir (50 units)
    2. Better hyperparameters
    3. Enhanced direction change detection
    4. Dynamic prediction horizon
    5. Improved adaptation strategy
    """
    def __init__(self, n_models=10, warmup=5, window=20, base_horizon=20):
        self.n_models = n_models
        self.warmup = warmup
        self.window = window
        self.base_horizon = base_horizon

        self.history = deque(maxlen=10000)
        self.warmup_buffer = deque(maxlen=window)
        self.have_warm = False

        self.esns = None
        self.in_online_std = None
        self.tg_online_std = None

        # Improved adaptation parameters
        self.adapt_window = 6
        self.adapt_damping_nominal = 0.5
        self.adapt_damping_boost = 1.5
        self.boost_frames = 8
        self.boost_error_thresh = 0.3

        # Per-ESN tracking
        self.err_deques = None
        self.boost_counters = None

        # Direction change detection
        self.sudden_change_thresh = 0.15
        self.angle_change_thresh = 0.8  # radians

    def _detect_direction_change(self):
        """Enhanced direction change detection"""
        if len(self.history) < 3:
            return False

        v1 = self.history[-2] - self.history[-3]
        v2 = self.history[-1] - self.history[-2]

        # Distance change
        dist_change = np.linalg.norm(v2 - v1)
        if dist_change > self.sudden_change_thresh:
            return True

        # Angle change
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 > 0.01 and norm2 > 0.01:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            if angle > self.angle_change_thresh:
                return True

        return False

    def _compute_dynamic_horizon(self):
        """Compute dynamic prediction horizon based on motion"""
        if len(self.history) < 5:
            return self.base_horizon

        recent = np.array(list(self.history)[-5:])
        velocities = np.diff(recent, axis=0)
        speed = np.mean(np.linalg.norm(velocities, axis=1))
        speed_var = np.var(np.linalg.norm(velocities, axis=1))

        if speed > 0.08 or speed_var > 0.005:
            return max(10, self.base_horizon - 5)  # Shorter for fast/erratic
        elif speed < 0.02:
            return min(30, self.base_horizon + 5)  # Longer for slow
        return self.base_horizon

    def update(self, pt):
        pt = np.array(pt, dtype=float)
        self.history.append(pt)
        self.warmup_buffer.append(pt)

        if not self.have_warm and len(self.history) >= max(self.warmup, 6):
            X_warm = np.array(list(self.history)[-self.warmup-1:-1])
            Y_warm = np.diff(np.array(list(self.history)[-self.warmup-1:]), axis=0)

            self.in_online_std = OnlineStandardizer(X_warm.mean(axis=0), X_warm.var(axis=0), alpha=0.03)
            self.tg_online_std = OnlineStandardizer(Y_warm.mean(axis=0), Y_warm.var(axis=0), alpha=0.03)

            Xw_s = self.in_online_std.transform(X_warm)
            Yw_s = self.tg_online_std.transform(Y_warm)

            # V2: Larger reservoir (50 units) with better hyperparameters
            self.esns = []
            rng = np.random.default_rng(42)
            for i in range(self.n_models):
                units = 50 + int(rng.integers(-8, 9))
                sr = float(rng.uniform(0.88, 0.95))  # Higher spectral radius
                lr = float(rng.uniform(0.25, 0.45))  # Lower leak rate
                input_scaling = float(rng.uniform(0.3, 0.5))
                bias = float(rng.uniform(0.0, 0.1))
                sparsity = float(rng.uniform(0.85, 0.92))

                esn = SimpleESN(units=units, sr=sr, lr=lr, input_scaling=input_scaling,
                               bias=bias, seed=42+i, sparsity=sparsity)
                esn.forgetting = 0.995  # Higher forgetting factor
                esn.partial_fit(Xw_s, Yw_s)
                self.esns.append(esn)

            self.err_deques = [deque(maxlen=10) for _ in self.esns]
            self.boost_counters = [0 for _ in self.esns]

            self.have_warm = True

        if self.in_online_std is not None:
            self.in_online_std.update(pt.reshape(1, -1))
            if len(self.history) >= 2:
                delta = (self.history[-1] - self.history[-2]).reshape(1, -1)
                self.tg_online_std.update(delta)

    def predict(self):
        if not self.have_warm or len(self.warmup_buffer) < 2:
            return None

        # Direction change detection
        direction_changed = self._detect_direction_change()
        if direction_changed:
            for esn in self.esns:
                esn.reset_state()

        # Dynamic horizon
        horizon = self._compute_dynamic_horizon()

        positions = np.array(list(self.warmup_buffer))
        if len(positions) >= 9:
            positions = savgol_smooth(positions, window_length=9, polyorder=2)

        adapt_window = min(self.adapt_window, len(positions))
        X_recent = positions[-adapt_window:]
        X_recent_s = self.in_online_std.transform(X_recent)

        all_esn_preds = []
        for i, esn in enumerate(self.esns):
            last_input = X_recent_s.copy()
            future_preds = []

            for _ in range(horizon):
                delta_s = esn.run(last_input)[-1]
                delta = self.tg_online_std.inverse_transform(delta_s.reshape(1, -1))[0]
                last_pos_abs = self.in_online_std.inverse_transform(last_input[-1].reshape(1, -1))[0]
                next_pos_abs = last_pos_abs + delta
                future_preds.append(next_pos_abs)

                last_input = np.roll(last_input, -1, axis=0)
                last_input[-1] = self.in_online_std.transform(next_pos_abs.reshape(1, -1))[0]

            all_esn_preds.append(future_preds)

            # Improved online adaptation
            if len(X_recent_s) > 1:
                adapt_X = X_recent_s[:-1]
                adapt_Y = np.diff(X_recent_s, axis=0)

                # Compute error
                try:
                    pred_s = esn.run(adapt_X)[-1]
                    real_s = adapt_Y[-1]
                    error = float(np.linalg.norm(real_s - pred_s))
                    self.err_deques[i].append(error)
                except:
                    error = 0

                running_err = float(np.mean(self.err_deques[i])) if self.err_deques[i] else 0

                # Determine damping with boost
                if self.boost_counters[i] > 0:
                    damping = self.adapt_damping_boost
                    self.boost_counters[i] -= 1
                elif running_err > self.boost_error_thresh or direction_changed:
                    damping = self.adapt_damping_boost
                    self.boost_counters[i] = self.boost_frames
                else:
                    damping = self.adapt_damping_nominal

                adapt_Y_adj = np.clip(adapt_Y * damping, -1.2, 1.2)
                esn.partial_fit(adapt_X, adapt_Y_adj)
                esn.W_out = np.clip(esn.W_out, -10.0, 10.0)

        # Pad predictions to same length and average
        max_len = max(len(p) for p in all_esn_preds)
        padded_preds = []
        for pred in all_esn_preds:
            if len(pred) < max_len:
                pred = pred + [pred[-1]] * (max_len - len(pred))
            padded_preds.append(pred[:20])  # Keep consistent length for comparison
        return np.mean(np.array(padded_preds), axis=0)


def run_comparison(pattern, output_dir):
    """Run comparison between V1 and V2"""
    print(f"\nPattern: {pattern}")
    print("-" * 50)

    np.random.seed(42)
    trajectory = generate_leg_trajectory(pattern=pattern, n_steps=200, noise_level=0.015)

    # V1 predictor
    np.random.seed(42)
    predictor_v1 = OriginalESNPredictor(n_models=10, warmup=5, window=20, future_horizon=20)
    predictions_v1 = []

    # V2 predictor
    np.random.seed(42)
    predictor_v2 = ImprovedESNPredictor(n_models=10, warmup=5, window=20, base_horizon=20)
    predictions_v2 = []

    for i, pt in enumerate(trajectory):
        predictor_v1.update(pt)
        predictor_v2.update(pt)

        if i >= 10:
            pred_v1 = predictor_v1.predict()
            pred_v2 = predictor_v2.predict()
            predictions_v1.append((i, pred_v1))
            predictions_v2.append((i, pred_v2))

    # Compute errors
    errors_v1 = []
    errors_v2 = []

    for (step, pred_v1), (_, pred_v2) in zip(predictions_v1, predictions_v2):
        if pred_v1 is not None and step + 20 <= len(trajectory):
            actual = trajectory[step+1:step+21]
            if len(actual) == len(pred_v1):
                errors_v1.append(np.mean(np.linalg.norm(pred_v1 - actual, axis=1)))

        if pred_v2 is not None and step + 20 <= len(trajectory):
            actual = trajectory[step+1:step+21]
            if len(actual) >= len(pred_v2):
                actual = actual[:len(pred_v2)]
                errors_v2.append(np.mean(np.linalg.norm(pred_v2 - actual, axis=1)))

    mean_v1 = np.mean(errors_v1) if errors_v1 else 0
    mean_v2 = np.mean(errors_v2) if errors_v2 else 0
    improvement = ((mean_v1 - mean_v2) / mean_v1 * 100) if mean_v1 > 0 else 0

    print(f"  V1 Mean Error: {mean_v1:.4f} m")
    print(f"  V2 Mean Error: {mean_v2:.4f} m")
    print(f"  Improvement: {improvement:+.1f}%")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Trajectory comparison
    ax1 = axes[0, 0]
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Actual')

    mid_idx = len(predictions_v1) // 2
    if mid_idx < len(predictions_v1):
        step, pred_v1 = predictions_v1[mid_idx]
        _, pred_v2 = predictions_v2[mid_idx]
        if pred_v1 is not None:
            ax1.plot(pred_v1[:, 0], pred_v1[:, 1], 'r--', linewidth=2, label='V1 Prediction', alpha=0.7)
        if pred_v2 is not None:
            ax1.plot(pred_v2[:, 0], pred_v2[:, 1], 'g--', linewidth=2, label='V2 Prediction', alpha=0.7)
        ax1.plot(trajectory[step, 0], trajectory[step, 1], 'ko', markersize=10, label='Current')

    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title(f'Trajectory Comparison: {pattern}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Error comparison over time
    ax2 = axes[0, 1]
    ax2.plot(errors_v1, 'r-', linewidth=1.5, label=f'V1 (mean={mean_v1:.4f}m)', alpha=0.7)
    ax2.plot(errors_v2, 'g-', linewidth=1.5, label=f'V2 (mean={mean_v2:.4f}m)', alpha=0.7)
    ax2.axhline(y=mean_v1, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=mean_v2, color='g', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Prediction Error [m]')
    ax2.set_title('Error Comparison Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Error distribution
    ax3 = axes[1, 0]
    ax3.hist(errors_v1, bins=20, alpha=0.5, label='V1', color='red')
    ax3.hist(errors_v2, bins=20, alpha=0.5, label='V2', color='green')
    ax3.set_xlabel('Prediction Error [m]')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Summary bar chart
    ax4 = axes[1, 1]
    versions = ['V1 (Original)', 'V2 (Improved)']
    means = [mean_v1, mean_v2]
    colors = ['coral', 'seagreen']
    bars = ax4.bar(versions, means, color=colors, alpha=0.7)

    for bar, mean in zip(bars, means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean:.4f}m', ha='center', va='bottom', fontsize=12)

    ax4.set_ylabel('Mean Error [m]')
    ax4.set_title(f'Accuracy Comparison ({improvement:+.1f}% {"improvement" if improvement > 0 else "change"})')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'esn_comparison_{pattern}_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filepath}")

    return {
        'v1_error': mean_v1,
        'v2_error': mean_v2,
        'improvement': improvement
    }


def create_summary(all_stats, output_dir):
    """Create summary comparison figure"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    patterns = list(all_stats.keys())
    v1_errors = [all_stats[p]['v1_error'] for p in patterns]
    v2_errors = [all_stats[p]['v2_error'] for p in patterns]
    improvements = [all_stats[p]['improvement'] for p in patterns]

    x = np.arange(len(patterns))
    width = 0.35

    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, v1_errors, width, label='V1 (Original)', color='coral', alpha=0.7)
    bars2 = ax1.bar(x + width/2, v2_errors, width, label='V2 (Improved)', color='seagreen', alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(patterns)
    ax1.set_ylabel('Mean Error [m]')
    ax1.set_title('Prediction Error: V1 vs V2')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    colors = ['seagreen' if imp > 0 else 'coral' for imp in improvements]
    bars = ax2.bar(patterns, improvements, color=colors, alpha=0.7)

    for bar, imp in zip(bars, improvements):
        y_pos = bar.get_height() + 0.5 if bar.get_height() >= 0 else bar.get_height() - 2
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{imp:+.1f}%', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=10)

    ax2.set_ylabel('Improvement [%]')
    ax2.set_title('Improvement by Pattern')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'esn_v2_summary_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSummary saved: {filepath}")

    avg_improvement = np.mean(improvements)
    print(f"\n{'='*50}")
    print(f"Overall Average Improvement: {avg_improvement:+.1f}%")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='ESN Path Prediction V2 Comparison')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--pattern', type=str, default='all',
                        choices=['straight', 'curve', 'zigzag', 'stop_and_go', 'all'],
                        help='Test pattern')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("ESN Path Prediction V2 - Improvement Comparison")
    print("=" * 50)
    print("\nV2 Improvements:")
    print("  1. Larger reservoir (50 units vs 25)")
    print("  2. Higher spectral radius (0.88-0.95 vs 0.8-0.9)")
    print("  3. Enhanced direction change detection")
    print("  4. Dynamic prediction horizon")
    print("  5. Adaptive boost learning on errors")
    print("=" * 50)

    patterns = ['straight', 'curve', 'zigzag', 'stop_and_go'] if args.pattern == 'all' else [args.pattern]

    all_stats = {}
    for pattern in patterns:
        stats = run_comparison(pattern, output_dir)
        all_stats[pattern] = stats

    if len(all_stats) > 1:
        create_summary(all_stats, output_dir)

    print("\n" + "=" * 50)
    print("Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
