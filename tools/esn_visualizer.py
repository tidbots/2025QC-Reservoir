#!/usr/bin/env python3
"""
ESN Path Prediction Visualizer
==============================
Standalone script to test and visualize the ESN path prediction algorithm without ROS.

Usage:
    python3 esn_visualizer.py [--output OUTPUT_DIR] [--pattern PATTERN]

Patterns:
    - straight: Straight walking
    - curve: Curved walking
    - zigzag: Zigzag walking
    - stop_and_go: Stop and go
    - all: All patterns

Output:
    - PNG images (prediction visualization)
    - Statistics (prediction accuracy)
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
    """Simple Echo State Network implementation"""
    def __init__(self, units=25, sr=0.85, lr=0.5, input_scaling=0.3, bias=0.0, seed=42):
        self.units = units
        self.sr = sr
        self.lr = lr
        self.input_scaling = input_scaling
        self.bias = bias

        rng = np.random.default_rng(seed)

        # Input weights
        self.W_in = rng.uniform(-input_scaling, input_scaling, (units, 2))

        # Reservoir weights (sparse)
        W = rng.uniform(-1, 1, (units, units))
        W[rng.random((units, units)) > 0.1] = 0  # 90% sparse
        # Scale by spectral radius
        eigenvalues = np.linalg.eigvals(W)
        W = W * (sr / np.max(np.abs(eigenvalues) + 1e-6))
        self.W = W

        # Output weights (learned by RLS)
        self.W_out = np.zeros((2, units))

        # Reservoir state
        self.state = np.zeros(units)

        # RLS parameters
        self.P = np.eye(units) * 1000
        self.forgetting = 0.99

        # Bias
        self.b = np.full(units, bias)

    def run(self, X):
        """Run reservoir on input X and return output"""
        X = np.atleast_2d(X)
        outputs = []
        for x in X:
            # Update reservoir state
            pre_act = self.W_in @ x + self.W @ self.state + self.b
            self.state = (1 - self.lr) * self.state + self.lr * np.tanh(pre_act)
            # Compute output
            y = self.W_out @ self.state
            outputs.append(y)
        return np.array(outputs)

    def partial_fit(self, X, Y):
        """Online learning with RLS"""
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        for x, y in zip(X, Y):
            # Update reservoir state
            pre_act = self.W_in @ x + self.W @ self.state + self.b
            self.state = (1 - self.lr) * self.state + self.lr * np.tanh(pre_act)

            # RLS update
            k = self.P @ self.state / (self.forgetting + self.state @ self.P @ self.state)
            e = y - self.W_out @ self.state
            self.W_out = self.W_out + np.outer(e, k)
            self.P = (self.P - np.outer(k, self.state @ self.P)) / self.forgetting

    def reset_state(self):
        self.state = np.zeros(self.units)


def create_diverse_esns(n_models=10, base_units=25, seed=42):
    """Create diverse ESN ensemble"""
    esns = []
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        units = base_units + int(rng.integers(-5, 6))
        sr = float(rng.uniform(0.8, 0.9))
        lr = float(rng.uniform(0.35, 0.6))
        input_scaling = float(rng.uniform(0.2, 0.4))
        bias = float(rng.uniform(-0.2, 0.2))
        esn = SimpleESN(units=units, sr=sr, lr=lr, input_scaling=input_scaling, bias=bias, seed=seed+i)
        esns.append(esn)
    return esns


def savgol_win(win, window_length=9, polyorder=2):
    """Savitzky-Golay smoothing"""
    arr = np.asarray(win)
    if arr.shape[0] < window_length or window_length % 2 == 0:
        return arr
    x = savgol_filter(arr[:, 0], window_length, polyorder, mode="interp")
    y = savgol_filter(arr[:, 1], window_length, polyorder, mode="interp")
    return np.stack([x, y], axis=1)


def generate_leg_trajectory(pattern='straight', n_steps=200, noise_level=0.02):
    """Generate synthetic leg trajectory data"""
    t = np.linspace(0, 4 * np.pi, n_steps)

    if pattern == 'straight':
        # Straight walking (forward)
        x = np.linspace(0.5, 1.5, n_steps)
        y = np.zeros(n_steps) + 0.1 * np.sin(t * 2)  # Small lateral sway

    elif pattern == 'curve':
        # Curved walking (arc)
        theta = np.linspace(0, np.pi/2, n_steps)
        r = 1.0
        x = r * np.cos(theta) + 0.5
        y = r * np.sin(theta) - 0.5

    elif pattern == 'zigzag':
        # Zigzag walking
        x = np.linspace(0.5, 1.5, n_steps)
        y = 0.3 * np.sin(t * 1.5)

    elif pattern == 'stop_and_go':
        # Stop and go
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        for i in range(n_steps):
            if i < n_steps // 4:
                x[i] = 0.5 + i * 0.004
            elif i < n_steps // 2:
                x[i] = x[n_steps // 4 - 1]  # Stop
            elif i < 3 * n_steps // 4:
                x[i] = x[n_steps // 2 - 1] + (i - n_steps // 2) * 0.004
            else:
                x[i] = x[3 * n_steps // 4 - 1]  # Stop
        y = 0.05 * np.sin(t)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Add noise
    x += np.random.randn(n_steps) * noise_level
    y += np.random.randn(n_steps) * noise_level

    return np.stack([x, y], axis=1)


class ESNPredictor:
    """ESN Predictor (standalone version without ROS)"""
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

        # Adaptation parameters
        self.adapt_window = 5
        self.adapt_damping = 0.35
        self.state_clip = 5.0
        self.wout_clip = 8.0

    def update(self, pt):
        """Update with new position"""
        pt = np.array(pt, dtype=float)
        self.history.append(pt)
        self.warmup_buffer.append(pt)

        # Check warmup completion
        if not self.have_warm and len(self.history) >= max(self.warmup, 6):
            X_warm = np.array(list(self.history)[-self.warmup-1:-1])
            Y_warm = np.diff(np.array(list(self.history)[-self.warmup-1:]), axis=0)

            self.in_online_std = OnlineStandardizer(X_warm.mean(axis=0), X_warm.var(axis=0))
            self.tg_online_std = OnlineStandardizer(Y_warm.mean(axis=0), Y_warm.var(axis=0))

            Xw_s = self.in_online_std.transform(X_warm)
            Yw_s = self.tg_online_std.transform(Y_warm)

            self.esns = create_diverse_esns(n_models=self.n_models)
            for esn in self.esns:
                esn.partial_fit(Xw_s, Yw_s)

            self.have_warm = True

        # Update online standardizers
        if self.in_online_std is not None:
            self.in_online_std.update(pt.reshape(1, -1))
            if len(self.history) >= 2:
                delta = (self.history[-1] - self.history[-2]).reshape(1, -1)
                self.tg_online_std.update(delta)

    def predict(self):
        """Predict future path"""
        if not self.have_warm or len(self.warmup_buffer) < 2:
            return None

        # Smoothing
        smoothed = savgol_win(list(self.warmup_buffer), window_length=9, polyorder=2)
        adapt_window = min(self.adapt_window, len(smoothed))
        X_recent = smoothed[-adapt_window:]
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
                esn.W_out = np.clip(esn.W_out, -self.wout_clip, self.wout_clip)

        return np.mean(np.array(all_esn_preds), axis=0)


def visualize_prediction(trajectory, predictions_history, pattern_name, output_dir):
    """Visualize prediction results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Overall trajectory and predictions
    ax1 = axes[0, 0]
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Actual Trajectory')

    # Show multiple predictions (thinned out)
    for i, (step, pred) in enumerate(predictions_history[::10]):
        if pred is not None:
            color = plt.cm.Reds(0.3 + 0.7 * i / (len(predictions_history[::10])))
            ax1.plot(pred[:, 0], pred[:, 1], '-', color=color, alpha=0.5, linewidth=1)

    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title(f'Trajectory and Predictions: {pattern_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Prediction detail at specific time point
    ax2 = axes[0, 1]
    mid_idx = len(predictions_history) // 2
    if mid_idx < len(predictions_history):
        step, pred = predictions_history[mid_idx]
        if pred is not None:
            # Past trajectory
            past = trajectory[:step+1]
            future_actual = trajectory[step+1:step+21] if step+21 <= len(trajectory) else trajectory[step+1:]

            ax2.plot(past[:, 0], past[:, 1], 'b-', linewidth=2, label='Past Trajectory')
            ax2.plot(past[-1, 0], past[-1, 1], 'bo', markersize=10, label='Current Position')
            ax2.plot(future_actual[:, 0], future_actual[:, 1], 'g-', linewidth=2, label='Actual Future')
            ax2.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='ESN Prediction')
            ax2.plot(pred[-1, 0], pred[-1, 1], 'r^', markersize=8)

            ax2.set_xlabel('X [m]')
            ax2.set_ylabel('Y [m]')
            ax2.set_title(f'Prediction Detail at Step {step}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')

    # 3. Prediction error time series
    ax3 = axes[1, 0]
    errors = []
    steps = []
    for step, pred in predictions_history:
        if pred is not None and step + 20 <= len(trajectory):
            actual_future = trajectory[step+1:step+21]
            if len(actual_future) == len(pred):
                error = np.mean(np.linalg.norm(pred - actual_future, axis=1))
                errors.append(error)
                steps.append(step)

    if errors:
        ax3.plot(steps, errors, 'b-', linewidth=1.5)
        ax3.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean Error: {np.mean(errors):.4f} m')
        ax3.fill_between(steps, 0, errors, alpha=0.3)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Prediction Error [m]')
        ax3.set_title('Prediction Error Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. X/Y component prediction accuracy
    ax4 = axes[1, 1]
    x_errors = []
    y_errors = []
    for step, pred in predictions_history:
        if pred is not None and step + 20 <= len(trajectory):
            actual_future = trajectory[step+1:step+21]
            if len(actual_future) == len(pred):
                x_err = np.mean(np.abs(pred[:, 0] - actual_future[:, 0]))
                y_err = np.mean(np.abs(pred[:, 1] - actual_future[:, 1]))
                x_errors.append(x_err)
                y_errors.append(y_err)

    if x_errors:
        ax4.bar(['X Component', 'Y Component'], [np.mean(x_errors), np.mean(y_errors)],
                color=['steelblue', 'coral'], alpha=0.7)
        ax4.set_ylabel('Mean Absolute Error [m]')
        ax4.set_title('Prediction Accuracy by Component')
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'esn_prediction_{pattern_name}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filepath}")

    # Return statistics
    if errors:
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors)
        }
    return None


def run_simulation(pattern, output_dir):
    """Run simulation"""
    print(f"\nPattern: {pattern}")
    print("-" * 40)

    # Generate trajectory
    trajectory = generate_leg_trajectory(pattern=pattern, n_steps=200, noise_level=0.015)
    print(f"  Trajectory generated: {len(trajectory)} steps")

    # ESN predictor
    predictor = ESNPredictor(n_models=10, warmup=5, window=20, future_horizon=20)

    # Run simulation
    predictions_history = []
    for i, pt in enumerate(trajectory):
        predictor.update(pt)

        if i >= 10:  # After warmup
            pred = predictor.predict()
            predictions_history.append((i, pred))

    print(f"  Predictions made: {len(predictions_history)}")

    # Visualize
    stats = visualize_prediction(trajectory, predictions_history, pattern, output_dir)

    if stats:
        print(f"  Mean Error: {stats['mean_error']:.4f} m")
        print(f"  Std Dev: {stats['std_error']:.4f} m")

    return stats


def create_summary_figure(all_stats, output_dir):
    """Create summary comparison figure for all patterns"""
    fig, ax = plt.subplots(figsize=(10, 6))

    patterns = list(all_stats.keys())
    means = [all_stats[p]['mean_error'] for p in patterns]
    stds = [all_stats[p]['std_error'] for p in patterns]

    x = np.arange(len(patterns))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'coral', 'seagreen', 'gold'], alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.set_ylabel('Prediction Error [m]')
    ax.set_title('ESN Path Prediction: Accuracy Comparison by Pattern')
    ax.grid(True, alpha=0.3, axis='y')

    # Display values on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'esn_summary_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSummary saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='ESN Path Prediction Visualizer')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--pattern', type=str, default='all',
                        choices=['straight', 'curve', 'zigzag', 'stop_and_go', 'all'],
                        help='Test pattern')
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("ESN Path Prediction Visualizer")
    print("=" * 50)

    patterns = ['straight', 'curve', 'zigzag', 'stop_and_go'] if args.pattern == 'all' else [args.pattern]

    all_stats = {}
    for pattern in patterns:
        stats = run_simulation(pattern, output_dir)
        if stats:
            all_stats[pattern] = stats

    # Summary for all patterns
    if len(all_stats) > 1:
        create_summary_figure(all_stats, output_dir)

    print("\n" + "=" * 50)
    print("Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
