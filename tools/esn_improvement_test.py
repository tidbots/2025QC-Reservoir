#!/usr/bin/env python3
"""
ESN Improvement Test - Incremental Validation
==============================================
Tests improvements one at a time:
1. Direction change detection tuning
2. Kalman filter hybridization
3. Combined approach

Usage:
    python3 esn_improvement_test.py --output output
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
    def __init__(self, units=25, sr=0.85, lr=0.5, input_scaling=0.3, bias=0.0, seed=42):
        self.units = units
        rng = np.random.default_rng(seed)

        self.W_in = rng.uniform(-input_scaling, input_scaling, (units, 2))

        W = rng.uniform(-1, 1, (units, units))
        W[rng.random((units, units)) > 0.1] = 0
        eigenvalues = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0:
            W = W * (sr / max_eig)
        self.W = W

        self.W_out = np.zeros((2, units))
        self.state = np.zeros(units)
        self.P = np.eye(units) * 1000
        self.forgetting = 0.99
        self.lr = lr
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


class SimpleKalmanFilter:
    """
    Simple Kalman Filter for 2D position tracking
    State: [x, y, vx, vy]
    """
    def __init__(self, dt=0.1, process_noise=0.1, measurement_noise=0.05):
        self.dt = dt

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance
        self.Q = np.eye(4) * process_noise

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise

        # State estimate
        self.x = np.zeros(4)

        # Estimate covariance
        self.P = np.eye(4)

        self.initialized = False

    def initialize(self, pos):
        """Initialize with first position"""
        self.x = np.array([pos[0], pos[1], 0, 0])
        self.P = np.eye(4)
        self.initialized = True

    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement):
        """Update with measurement"""
        if not self.initialized:
            self.initialize(measurement)
            return measurement

        # Predict
        self.predict()

        # Update
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[:2].copy()

    def get_velocity(self):
        """Get current velocity estimate"""
        return self.x[2:4].copy()

    def predict_future(self, steps):
        """Predict future positions"""
        positions = []
        x_temp = self.x.copy()
        for _ in range(steps):
            x_temp = self.F @ x_temp
            positions.append(x_temp[:2].copy())
        return np.array(positions)


def savgol_smooth(data, window_length=9, polyorder=2):
    arr = np.asarray(data)
    if arr.shape[0] < window_length:
        return arr
    smoothed = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        smoothed[:, i] = savgol_filter(arr[:, i], window_length, polyorder, mode="interp")
    return smoothed


def generate_leg_trajectory(pattern='straight', n_steps=200, noise_level=0.02):
    t = np.linspace(0, 4 * np.pi, n_steps)
    if pattern == 'straight':
        x = np.linspace(0.5, 1.5, n_steps)
        y = np.zeros(n_steps) + 0.1 * np.sin(t * 2)
    elif pattern == 'curve':
        theta = np.linspace(0, np.pi/2, n_steps)
        x = np.cos(theta) + 0.5
        y = np.sin(theta) - 0.5
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
    x += np.random.randn(n_steps) * noise_level
    y += np.random.randn(n_steps) * noise_level
    return np.stack([x, y], axis=1)


# ============ V1: Original Predictor ============

class OriginalESNPredictor:
    """Original ESN Predictor (baseline)"""
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

            if len(X_recent_s) > 1:
                adapt_X = X_recent_s[:-1]
                adapt_Y = np.diff(X_recent_s, axis=0) * self.adapt_damping
                esn.partial_fit(adapt_X, adapt_Y)
                esn.W_out = np.clip(esn.W_out, -8.0, 8.0)

        return np.mean(np.array(all_esn_preds), axis=0)


# ============ Improvement 1: Direction Change Detection ============

class DirectionTunedESNPredictor:
    """ESN with tuned direction change detection only"""
    def __init__(self, n_models=10, warmup=5, window=20, future_horizon=20,
                 angle_thresh=0.5, speed_thresh=0.1):
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

        # Tunable thresholds
        self.angle_thresh = angle_thresh
        self.speed_thresh = speed_thresh

    def _detect_direction_change(self):
        """Detect direction change with tuned thresholds"""
        if len(self.history) < 3:
            return False

        v1 = self.history[-2] - self.history[-3]
        v2 = self.history[-1] - self.history[-2]

        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)

        # Speed change detection
        if abs(norm2 - norm1) > self.speed_thresh:
            return True

        # Angle change detection
        if norm1 > 0.005 and norm2 > 0.005:
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
            angle = np.arccos(cos_angle)
            if angle > self.angle_thresh:
                return True

        return False

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

        # Direction change detection - reset ESN states
        if self._detect_direction_change():
            for esn in self.esns:
                esn.reset_state()

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

            if len(X_recent_s) > 1:
                adapt_X = X_recent_s[:-1]
                adapt_Y = np.diff(X_recent_s, axis=0) * self.adapt_damping
                esn.partial_fit(adapt_X, adapt_Y)
                esn.W_out = np.clip(esn.W_out, -8.0, 8.0)

        return np.mean(np.array(all_esn_preds), axis=0)


# ============ Improvement 2: Kalman Filter Hybrid ============

class KalmanHybridESNPredictor:
    """ESN + Kalman Filter hybrid predictor"""
    def __init__(self, n_models=10, warmup=5, window=20, future_horizon=20,
                 kalman_weight=0.3):
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

        # Kalman filter
        self.kalman = SimpleKalmanFilter(dt=0.1, process_noise=0.05, measurement_noise=0.02)
        self.kalman_weight = kalman_weight  # Weight for Kalman predictions

    def update(self, pt):
        pt = np.array(pt, dtype=float)
        self.history.append(pt)
        self.warmup_buffer.append(pt)

        # Update Kalman filter
        self.kalman.update(pt)

        if not self.have_warm and len(self.history) >= max(self.warmup, 6):
            X_warm = np.array(list(self.history)[-self.warmup-1:-1])
            Y_warm = np.diff(np.array(list(self.history)[-self.warmup-1:]), axis=0)
            self.in_online_std = OnlineStandardizer(X_warm.mean(axis=0), X_warm.var(axis=0))
            self.tg_online_std = OnlineStandardizer(Y_warm.mean(axis=0), Y_warm.var(axis=0))
            Xw_s = self.in_online_std.transform(X_warm)
            Yw_s = self.tg_online_std.transform(Y_warm)

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

        # ESN predictions
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

            if len(X_recent_s) > 1:
                adapt_X = X_recent_s[:-1]
                adapt_Y = np.diff(X_recent_s, axis=0) * self.adapt_damping
                esn.partial_fit(adapt_X, adapt_Y)
                esn.W_out = np.clip(esn.W_out, -8.0, 8.0)

        esn_pred = np.mean(np.array(all_esn_preds), axis=0)

        # Kalman filter predictions
        kalman_pred = self.kalman.predict_future(self.future_horizon)

        # Weighted combination
        combined_pred = (1 - self.kalman_weight) * esn_pred + self.kalman_weight * kalman_pred

        return combined_pred


# ============ Improvement 3: Combined Approach ============

class CombinedESNPredictor:
    """ESN with direction detection + Kalman hybrid"""
    def __init__(self, n_models=10, warmup=5, window=20, future_horizon=20,
                 angle_thresh=0.5, speed_thresh=0.1, kalman_weight=0.3):
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

        # Direction detection
        self.angle_thresh = angle_thresh
        self.speed_thresh = speed_thresh

        # Kalman filter
        self.kalman = SimpleKalmanFilter(dt=0.1, process_noise=0.05, measurement_noise=0.02)
        self.kalman_weight = kalman_weight

    def _detect_direction_change(self):
        if len(self.history) < 3:
            return False
        v1 = self.history[-2] - self.history[-3]
        v2 = self.history[-1] - self.history[-2]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if abs(norm2 - norm1) > self.speed_thresh:
            return True
        if norm1 > 0.005 and norm2 > 0.005:
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
            if np.arccos(cos_angle) > self.angle_thresh:
                return True
        return False

    def update(self, pt):
        pt = np.array(pt, dtype=float)
        self.history.append(pt)
        self.warmup_buffer.append(pt)
        self.kalman.update(pt)

        if not self.have_warm and len(self.history) >= max(self.warmup, 6):
            X_warm = np.array(list(self.history)[-self.warmup-1:-1])
            Y_warm = np.diff(np.array(list(self.history)[-self.warmup-1:]), axis=0)
            self.in_online_std = OnlineStandardizer(X_warm.mean(axis=0), X_warm.var(axis=0))
            self.tg_online_std = OnlineStandardizer(Y_warm.mean(axis=0), Y_warm.var(axis=0))
            Xw_s = self.in_online_std.transform(X_warm)
            Yw_s = self.tg_online_std.transform(Y_warm)

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

        if self._detect_direction_change():
            for esn in self.esns:
                esn.reset_state()

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

            if len(X_recent_s) > 1:
                adapt_X = X_recent_s[:-1]
                adapt_Y = np.diff(X_recent_s, axis=0) * self.adapt_damping
                esn.partial_fit(adapt_X, adapt_Y)
                esn.W_out = np.clip(esn.W_out, -8.0, 8.0)

        esn_pred = np.mean(np.array(all_esn_preds), axis=0)
        kalman_pred = self.kalman.predict_future(self.future_horizon)

        return (1 - self.kalman_weight) * esn_pred + self.kalman_weight * kalman_pred


def evaluate_predictor(predictor_class, pattern, **kwargs):
    """Evaluate a predictor on a pattern"""
    np.random.seed(42)
    trajectory = generate_leg_trajectory(pattern=pattern, n_steps=200, noise_level=0.015)

    np.random.seed(42)
    predictor = predictor_class(**kwargs)

    errors = []
    for i, pt in enumerate(trajectory):
        predictor.update(pt)
        if i >= 10:
            pred = predictor.predict()
            if pred is not None and i + 20 <= len(trajectory):
                actual = trajectory[i+1:i+21]
                if len(actual) == len(pred):
                    errors.append(np.mean(np.linalg.norm(pred - actual, axis=1)))

    return np.mean(errors) if errors else 0


def run_all_tests(output_dir):
    """Run all improvement tests"""
    patterns = ['straight', 'curve', 'zigzag', 'stop_and_go']

    results = {
        'V1 (Baseline)': {},
        'Direction Tuned': {},
        'Kalman Hybrid': {},
        'Combined': {}
    }

    print("\n" + "=" * 60)
    print("Testing improvements on each pattern...")
    print("=" * 60)

    for pattern in patterns:
        print(f"\n{pattern}:")

        # V1 Baseline
        err = evaluate_predictor(OriginalESNPredictor, pattern, n_models=10)
        results['V1 (Baseline)'][pattern] = err
        print(f"  V1 (Baseline):     {err:.4f} m")

        # Direction Tuned
        err = evaluate_predictor(DirectionTunedESNPredictor, pattern, n_models=10,
                                angle_thresh=0.5, speed_thresh=0.08)
        results['Direction Tuned'][pattern] = err
        print(f"  Direction Tuned:   {err:.4f} m")

        # Kalman Hybrid
        err = evaluate_predictor(KalmanHybridESNPredictor, pattern, n_models=10,
                                kalman_weight=0.3)
        results['Kalman Hybrid'][pattern] = err
        print(f"  Kalman Hybrid:     {err:.4f} m")

        # Combined
        err = evaluate_predictor(CombinedESNPredictor, pattern, n_models=10,
                                angle_thresh=0.5, speed_thresh=0.08, kalman_weight=0.3)
        results['Combined'][pattern] = err
        print(f"  Combined:          {err:.4f} m")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = list(results.keys())
    x = np.arange(len(patterns))
    width = 0.2
    colors = ['coral', 'steelblue', 'seagreen', 'gold']

    ax1 = axes[0]
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [results[method][p] for p in patterns]
        ax1.bar(x + i * width, values, width, label=method, color=color, alpha=0.7)

    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(patterns)
    ax1.set_ylabel('Mean Error [m]')
    ax1.set_title('Prediction Error by Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Improvement percentages
    ax2 = axes[1]
    improvements = {}
    for method in methods[1:]:
        improvements[method] = []
        for pattern in patterns:
            baseline = results['V1 (Baseline)'][pattern]
            current = results[method][pattern]
            imp = ((baseline - current) / baseline * 100) if baseline > 0 else 0
            improvements[method].append(imp)

    x2 = np.arange(len(patterns))
    for i, (method, color) in enumerate(zip(methods[1:], colors[1:])):
        ax2.bar(x2 + i * width, improvements[method], width, label=method, color=color, alpha=0.7)

    ax2.set_xticks(x2 + width)
    ax2.set_xticklabels(patterns)
    ax2.set_ylabel('Improvement [%]')
    ax2.set_title('Improvement vs Baseline')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'esn_improvements_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved: {filepath}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Average improvement vs baseline:")
    print("=" * 60)
    for method in methods[1:]:
        avg_imp = np.mean(improvements[method])
        print(f"  {method}: {avg_imp:+.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='ESN Improvement Test')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ESN Improvement Test - Incremental Validation")
    print("=" * 60)
    print("\nTesting:")
    print("  1. Direction change detection tuning")
    print("  2. Kalman filter hybrid")
    print("  3. Combined approach")

    run_all_tests(output_dir)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
