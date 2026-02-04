#!/usr/bin/env python3
"""
LSM (Liquid State Machine) vs ESN Trajectory Prediction Comparison
Implements a simple LSM using Leaky Integrate-and-Fire neurons.
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
from sklearn.linear_model import Ridge
from reservoirpy.nodes import Reservoir, RLS
from datetime import datetime
import argparse


class SimpleLSM:
    """
    Simple Liquid State Machine with Leaky Integrate-and-Fire neurons.
    Uses rate-coded input/output for trajectory prediction.
    """
    def __init__(self, n_inputs=2, n_reservoir=100, n_outputs=2,
                 connectivity=0.1, spectral_radius=0.9,
                 tau=20.0, threshold=1.0, reset=0.0,
                 input_scaling=0.5, seed=42):
        """
        Args:
            n_inputs: Input dimension
            n_reservoir: Number of reservoir neurons
            n_outputs: Output dimension
            connectivity: Reservoir connectivity (sparse)
            spectral_radius: Spectral radius of reservoir weights
            tau: Membrane time constant (ms)
            threshold: Spike threshold
            reset: Reset potential after spike
            input_scaling: Input weight scaling
            seed: Random seed
        """
        np.random.seed(seed)

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.tau = tau
        self.threshold = threshold
        self.reset = reset

        # Input weights (dense)
        self.W_in = np.random.randn(n_reservoir, n_inputs) * input_scaling

        # Reservoir weights (sparse, random)
        W = np.random.randn(n_reservoir, n_reservoir)
        mask = np.random.rand(n_reservoir, n_reservoir) < connectivity
        W = W * mask

        # Scale to spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            W = W * (spectral_radius / max_eigenvalue)
        self.W_res = W

        # Membrane potentials
        self.membrane = np.zeros(n_reservoir)

        # Spike history for readout
        self.spike_history = deque(maxlen=10)

        # Readout weights (trained)
        self.W_out = None
        self.readout = Ridge(alpha=1.0)

        # State history for training
        self.states = []
        self.targets = []

    def reset_state(self):
        """Reset membrane potentials."""
        self.membrane = np.zeros(self.n_reservoir)
        self.spike_history.clear()

    def step(self, input_signal, dt=1.0):
        """
        Single time step of LSM.

        Args:
            input_signal: Input vector (n_inputs,)
            dt: Time step (ms)

        Returns:
            state: Reservoir state (for readout)
        """
        # Input current
        I_in = self.W_in @ input_signal

        # Recurrent current from reservoir
        # Use firing rate approximation from recent spikes
        if len(self.spike_history) > 0:
            recent_spikes = np.mean(list(self.spike_history), axis=0)
        else:
            recent_spikes = np.zeros(self.n_reservoir)
        I_rec = self.W_res @ recent_spikes

        # Leaky integrate
        self.membrane = self.membrane * (1 - dt / self.tau) + (I_in + I_rec) * dt / self.tau

        # Check for spikes
        spikes = (self.membrane >= self.threshold).astype(float)

        # Reset spiking neurons
        self.membrane[spikes > 0] = self.reset

        # Store spike pattern
        self.spike_history.append(spikes)

        # Return state: combination of membrane potential and spike rate
        if len(self.spike_history) > 0:
            spike_rate = np.mean(list(self.spike_history), axis=0)
        else:
            spike_rate = np.zeros(self.n_reservoir)

        # Concatenate membrane and spike rate for richer state
        state = np.concatenate([self.membrane / self.threshold, spike_rate])

        return state

    def partial_fit(self, X, Y):
        """
        Online training of readout weights.

        Args:
            X: Input sequence (T, n_inputs)
            Y: Target sequence (T, n_outputs)
        """
        # Collect states
        states = []
        for x in X:
            state = self.step(x)
            states.append(state)
        states = np.array(states)

        # Store for batch training
        self.states.extend(states)
        self.targets.extend(Y)

        # Fit readout if enough data
        if len(self.states) >= 10:
            self.readout.fit(np.array(self.states), np.array(self.targets))

    def run(self, X):
        """
        Run LSM and predict.

        Args:
            X: Input sequence (T, n_inputs)

        Returns:
            predictions: Output sequence (T, n_outputs)
        """
        states = []
        for x in X:
            state = self.step(x)
            states.append(state)
        states = np.array(states)

        if self.readout is not None and hasattr(self.readout, 'coef_'):
            return self.readout.predict(states)
        else:
            # Return zeros if not trained
            return np.zeros((len(X), self.n_outputs))


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


def create_esn(units=50, seed=42):
    """Create ESN for comparison."""
    from reservoirpy.nodes import Reservoir, RLS
    reservoir = Reservoir(units=units, sr=0.95, lr=0.6,
                         input_scaling=0.5, seed=seed)
    readout = RLS(forgetting=0.99)
    return reservoir >> readout


def generate_test_trajectory(traj_type, n_points=300, noise=0.01):
    """Generate test trajectory."""
    np.random.seed(42)

    if traj_type == "linear":
        t = np.linspace(0, 1, n_points)
        x = t * 10
        y = t * 5

    elif traj_type == "circle":
        t = np.linspace(0, 2 * np.pi, n_points)
        x = 5 * np.cos(t)
        y = 5 * np.sin(t)

    elif traj_type == "lorenz":
        dt = 0.01
        sigma, rho, beta = 10, 28, 8/3
        x_l = np.zeros(n_points)
        y_l = np.zeros(n_points)
        z_l = np.zeros(n_points)
        x_l[0], y_l[0], z_l[0] = 1, 1, 1
        for i in range(1, n_points):
            dx = sigma * (y_l[i-1] - x_l[i-1])
            dy = x_l[i-1] * (rho - z_l[i-1]) - y_l[i-1]
            dz = x_l[i-1] * y_l[i-1] - beta * z_l[i-1]
            x_l[i] = x_l[i-1] + dx * dt
            y_l[i] = y_l[i-1] + dy * dt
            z_l[i] = z_l[i-1] + dz * dt
        x = x_l / 10
        y = y_l / 10

    elif traj_type == "pendulum":
        dt = 0.05
        g, L = 9.8, 1.0
        theta = np.zeros(n_points)
        omega = np.zeros(n_points)
        theta[0] = np.pi * 0.8
        for i in range(1, n_points):
            omega[i] = omega[i-1] - (g/L) * np.sin(theta[i-1]) * dt
            theta[i] = theta[i-1] + omega[i] * dt
        x = L * np.sin(theta) * 5
        y = -L * np.cos(theta) * 5 + 5

    elif traj_type == "random_walk":
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        vx, vy = 0.05, 0.02
        for i in range(1, n_points):
            if np.random.random() < 0.1:
                angle = np.random.uniform(-np.pi, np.pi)
                speed = np.sqrt(vx**2 + vy**2)
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)
            x[i] = x[i-1] + vx
            y[i] = y[i-1] + vy
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    x += np.random.randn(n_points) * noise
    y += np.random.randn(n_points) * noise

    return np.column_stack([x, y])


def evaluate_methods(traj, warmup=10, window=20, future_horizon=20, seed=42):
    """Evaluate ESN, LSM, and Kalman on a trajectory."""

    if len(traj) < warmup + future_horizon + 10:
        return None

    # Initialize models
    esn = create_esn(units=50, seed=seed)
    lsm = SimpleLSM(n_inputs=2, n_reservoir=100, n_outputs=2,
                    connectivity=0.1, spectral_radius=0.9, seed=seed)
    kalman = SimpleKalmanFilter(dt=0.1)

    # Warmup data
    warmup_buffer = traj[:warmup]
    X_warm = warmup_buffer[:-1]
    Y_warm = warmup_buffer[1:] - warmup_buffer[:-1]  # Deltas

    # Standardizers
    scaler_in = StandardScaler().fit(X_warm)
    scaler_tg = StandardScaler().fit(Y_warm)

    in_std = OnlineStandardizer(scaler_in.mean_, scaler_in.var_, alpha=0.02)
    tg_std = OnlineStandardizer(scaler_tg.mean_, scaler_tg.var_, alpha=0.02)

    X_w_s = in_std.transform(X_warm)
    Y_w_s = tg_std.transform(Y_warm)

    # Train ESN and LSM on warmup
    esn.partial_fit(X_w_s, Y_w_s)
    lsm.partial_fit(X_w_s, Y_w_s)

    # Initialize Kalman
    for pos in warmup_buffer:
        kalman.update(pos)

    # Evaluation
    errors = {'esn': [], 'lsm': [], 'kalman': [], 'linear': []}

    window_buffer = list(warmup_buffer)

    for frame_idx in range(warmup, len(traj) - future_horizon):
        pos = traj[frame_idx]
        window_buffer.append(pos)
        if len(window_buffer) > window:
            window_buffer.pop(0)

        kalman.update(pos)
        in_std.update(pos.reshape(1, -1))

        # Recent input
        X_recent = np.array(window_buffer[-5:])
        X_recent_s = in_std.transform(X_recent)

        # ESN prediction
        try:
            esn_delta_s = esn.run(X_recent_s)[-1]
            esn_delta = tg_std.inverse_transform(esn_delta_s.reshape(1, -1))[0]
            esn_pred = pos + esn_delta * future_horizon
        except:
            esn_pred = pos

        # LSM prediction
        try:
            lsm_delta_s = lsm.run(X_recent_s)[-1]
            lsm_delta = tg_std.inverse_transform(lsm_delta_s.reshape(1, -1))[0]
            lsm_pred = pos + lsm_delta * future_horizon
        except:
            lsm_pred = pos

        # Kalman prediction
        kalman_pred = kalman.predict_future(future_horizon)[-1]

        # Linear extrapolation
        if len(window_buffer) >= 2:
            velocity = np.array(window_buffer[-1]) - np.array(window_buffer[-2])
            linear_pred = pos + velocity * future_horizon
        else:
            linear_pred = pos

        # Online adaptation
        if len(window_buffer) >= 3:
            adapt_X = in_std.transform(np.array(window_buffer[-3:-1]))
            adapt_Y = np.diff(in_std.transform(np.array(window_buffer[-3:])), axis=0)
            try:
                esn.partial_fit(adapt_X, adapt_Y * 0.35)
            except:
                pass
            try:
                lsm.partial_fit(adapt_X, adapt_Y * 0.35)
            except:
                pass

        if len(window_buffer) >= 2:
            last_delta = np.array(window_buffer[-1]) - np.array(window_buffer[-2])
            tg_std.update(last_delta.reshape(1, -1))

        # Ground truth
        gt_pos = traj[frame_idx + future_horizon]

        # Calculate errors
        errors['esn'].append(np.linalg.norm(esn_pred - gt_pos))
        errors['lsm'].append(np.linalg.norm(lsm_pred - gt_pos))
        errors['kalman'].append(np.linalg.norm(kalman_pred - gt_pos))
        errors['linear'].append(np.linalg.norm(linear_pred - gt_pos))

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/lsm')
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--future_horizon', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    trajectory_types = [
        ("linear", "Linear"),
        ("circle", "Circle"),
        ("lorenz", "Lorenz (Chaotic)"),
        ("pendulum", "Nonlinear Pendulum"),
        ("random_walk", "Random Walk"),
    ]

    print("=" * 80)
    print("LSM vs ESN vs Kalman Trajectory Prediction Comparison")
    print(f"Future Horizon: {args.future_horizon} steps")
    print("=" * 80)

    all_results = {}

    for traj_type, traj_name in trajectory_types:
        print(f"\n### {traj_name} ({traj_type}) ###")

        esn_all, lsm_all, kalman_all, linear_all = [], [], [], []

        for trial in range(args.n_trials):
            np.random.seed(trial * 100)
            trajectory = generate_test_trajectory(traj_type, n_points=300)
            errors = evaluate_methods(trajectory, seed=trial,
                                      future_horizon=args.future_horizon)

            if errors and len(errors['esn']) > 0:
                esn_all.append(np.mean(errors['esn']))
                lsm_all.append(np.mean(errors['lsm']))
                kalman_all.append(np.mean(errors['kalman']))
                linear_all.append(np.mean(errors['linear']))
                print(f"  Trial {trial+1}: Kalman={np.mean(errors['kalman']):.3f}, "
                      f"ESN={np.mean(errors['esn']):.3f}, "
                      f"LSM={np.mean(errors['lsm']):.3f}")

        if esn_all:
            all_results[traj_type] = {
                'name': traj_name,
                'esn': np.mean(esn_all),
                'lsm': np.mean(lsm_all),
                'kalman': np.mean(kalman_all),
                'linear': np.mean(linear_all)
            }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - LSM vs ESN vs Kalman")
    print("=" * 80)
    header = f"{'Trajectory':<20} {'Kalman':<10} {'ESN':<10} {'LSM':<10} {'Linear':<10} {'Best':<10}"
    print(header)
    print("-" * 80)

    for traj_type, data in all_results.items():
        methods = {'Kalman': data['kalman'], 'ESN': data['esn'],
                   'LSM': data['lsm'], 'Linear': data['linear']}
        best = min(methods, key=methods.get)

        print(f"{data['name']:<20} {data['kalman']:<10.3f} {data['esn']:<10.3f} "
              f"{data['lsm']:<10.3f} {data['linear']:<10.3f} {best:<10}")

    # ESN vs LSM comparison
    print("\n" + "=" * 80)
    print("ESN vs LSM Detailed Comparison")
    print("=" * 80)
    for traj_type, data in all_results.items():
        esn_err = data['esn']
        lsm_err = data['lsm']
        diff = (esn_err - lsm_err) / esn_err * 100 if esn_err > 0 else 0
        winner = "LSM" if lsm_err < esn_err else "ESN"
        print(f"{data['name']:<20}: ESN={esn_err:.3f}, LSM={lsm_err:.3f}, "
              f"diff={diff:+.1f}% [{winner} better]")

    # Visualization
    fig, axes = plt.subplots(1, len(all_results), figsize=(4*len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for idx, (traj_type, data) in enumerate(all_results.items()):
        ax = axes[idx]
        methods = ['Kalman', 'ESN', 'LSM', 'Linear']
        values = [data['kalman'], data['esn'], data['lsm'], data['linear']]
        min_val = min(values)
        colors = ['green' if v == min_val else 'steelblue' for v in values]

        bars = ax.bar(methods, values, color=colors)
        ax.set_title(data['name'])
        ax.set_ylabel('Mean Error')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('LSM vs ESN vs Kalman Comparison', fontsize=14)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f'lsm_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")


if __name__ == "__main__":
    main()
