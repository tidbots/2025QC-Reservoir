#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

import numpy as np
from collections import deque
from scipy.signal import savgol_filter

from reservoirpy.nodes import Reservoir, RLS
from sklearn.preprocessing import StandardScaler


class OnlineStandardizer:
    def __init__(self, mean, var, alpha=0.02):
        self.mean = np.array(mean, dtype=float)
        self.var = np.array(var, dtype=float)
        self.alpha = alpha
        self.eps = 1e-6
    def update(self, x):
        x = np.atleast_2d(x)
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x.mean(axis=0)
        self.var  = (1 - self.alpha) * self.var  + self.alpha * x.var(axis=0)
    def transform(self, x):  
        return (x - self.mean) / np.sqrt(self.var + self.eps)
    
    def inverse_transform(self, xs): 
        return xs * np.sqrt(self.var + self.eps) + self.mean

# ----------------------------
# Create multiple ESNs.
# ----------------------------
def create_diverse_esns(n_models=5, base_units=25, seed=42, rls_forgetting=0.99):
    esns = []
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        units = base_units + int(rng.integers(-5, 6))
        sr = float(rng.uniform(0.8, 0.9))
        lr = float(rng.uniform(0.35, 0.6))
        input_scaling = float(rng.uniform(0.2, 0.4))
        bias = float(rng.uniform(-0.2, 0.2))

        reservoir = Reservoir(units=units, 
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
# Savitzky–Golay smoothing
# ----------------------------
def savgol_win(win, window_length=9, polyorder=2):
    """Return a smoothed copy of the window as (N,2).
    If not enough samples, return the original values."""
    arr = np.asarray(win)  # (N, 2)
    if arr.shape[0] < window_length or window_length % 2 == 0:
        return arr
    x = savgol_filter(arr[:, 0], window_length, polyorder, mode="interp")
    y = savgol_filter(arr[:, 1], window_length, polyorder, mode="interp")
    return np.stack([x, y], axis=1)

# ----------------------------
# Online ESN with Ridge
# ----------------------------
class LegsESNPredictor(Node):
    def __init__(self):
        super().__init__('legs_esn_predictor')

        # Params
        self.declare_parameter('legs_topic',        '/hri/leg_finder/leg_pose')
        self.declare_parameter('frame_id',          'base_footprint')
        self.declare_parameter('warmup',            5)
        self.declare_parameter('window',            20)
        self.declare_parameter('future_horizon',    20)
        self.declare_parameter('n_models',          10)
        self.declare_parameter('leg_update_hz',     10.0)   # accept at most 10 msgs/sec
        self.declare_parameter('update_rate_hz',    20.0)  # publish predictions

        self.leg_update_hz  = self.get_parameter('leg_update_hz').get_parameter_value().double_value
        self.legs_topic     = self.get_parameter('legs_topic').get_parameter_value().string_value
        self.frame_id       = self.get_parameter('frame_id').get_parameter_value().string_value
        self.warmup         = self.get_parameter('warmup').get_parameter_value().integer_value
        self.window         = self.get_parameter('window').get_parameter_value().integer_value
        self.future_horizon = self.get_parameter('future_horizon').get_parameter_value().integer_value
        self.n_models       = self.get_parameter('n_models').get_parameter_value().integer_value
        rate_hz             = self.get_parameter('update_rate_hz').get_parameter_value().double_value

        # Pose filer: Savitzky–Golay config
        self.sg_window = 9        # must be odd; try 9, 11, or 7
        self.sg_poly   = 2        # 2 or 3 are common

        # Place to store the ESN-ready, smoothed recent slice
        self.X_recent_for_esn = None
        # =====

        # Buffers
        self.history = deque(maxlen=10000)
        self.warmup_buffer = deque(maxlen=self.window)
        self.have_warm = False

        # ESN setup (lazy after warmup scalers exist)
        self.esns = None
        self.in_online_std = None
        self.tg_online_std = None

        # Adaptation config
        self.adapt_window = 5
        self.delta_window = 6
        self.sudden_change_thresh = 0.6
        self.adapt_damping_nominal = 0.35
        self.adapt_damping_boost = 1.0
        self.boost_frames = 6
        self.boost_error_thresh = 0.5
        self.state_clip = 5.0
        self.wout_clip = 8.0
        self.err_clip = 1.0

        # runtime containers (filled after ESNs are created)
        self.err_deques = None
        self.boost_counters = None

        # Leg position update frequency
        self._min_dt_ns = int(1e9 / max(self.leg_update_hz, 0.1))  # guard divide-by-zero
        self._last_accept_time = None

        # ROS I/O
        self.sub_ = self.create_subscription(
            PointStamped, self.legs_topic, self.cb_leg, rclpy.qos.qos_profile_sensor_data)
        self.pub_path_ = self.create_publisher(Path, '/hri/leg_finder/predicted_path', 1)

        # Timer for publishing predictions
        self.processing_timer_ = self.create_timer(1.0 / rate_hz, self.leg_processing)

        self.get_logger().info('Legs ESN predictor running.')

    def cb_leg(self, msg: PointStamped):
        # basic sanity: frame consistency (optional warn)
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            # In production you might tf-transform; here we just warn once in a while
            pass

        # --- frequency gate (time-based) ---
        now = self.get_clock().now()
        if self._last_accept_time is not None:
            if (now - self._last_accept_time).nanoseconds < self._min_dt_ns:
                return
        self._last_accept_time = now

        pt = np.array([msg.point.x, msg.point.y], dtype=float)

        # --- C) Outlier gate (z-gate on recent window) ---
        #if len(self.warmup_buffer) >= 10:
        #    arr = np.asarray(self.warmup_buffer)
        #    mu = arr.mean(axis=0)
        #    sd = arr.std(axis=0) + 1e-6
        #    z = np.abs((pt - mu) / sd)
        #    if (z > 6.0).any():    # start with 6.0; tune to 5–8
        #        return             # drop this spike frame entirely
        # --------------------------------------------------

        self.history.append(pt)
        self.warmup_buffer.append(pt)

        # --- B) Savitzky–Golay smoothing over the window for ESN inputs ---
        smoothed = savgol_win(self.warmup_buffer, window_length=self.sg_window, polyorder=self.sg_poly)

        # Prepare the ESN's recent slice (do NOT mutate warmup_buffer)
        adapt_window = min(self.adapt_window, len(smoothed))
        self.X_recent_for_esn = smoothed[-adapt_window:] if adapt_window > 0 else None
        # ------------------------------------------------------------------

        # Initialize after minimal warmup
        if not self.have_warm and len(self.history) >= max(self.warmup, 6):

            # --- Preprocessing (fit on warmup)
            X_warm = np.array(list(self.history)[-self.warmup-1:-1])
            Y_warm = np.diff(np.array(list(self.history)[-self.warmup-1:]), axis=0)
            scaler_in = StandardScaler().fit(X_warm)
            scaler_tg = StandardScaler().fit(Y_warm)

            # --- Online-friendly standardizers (EWMA initialized from batch scalers)
            self.in_online_std = OnlineStandardizer(scaler_in.mean_, scaler_in.var_, alpha=0.02)
            self.tg_online_std = OnlineStandardizer(scaler_tg.mean_, scaler_tg.var_, alpha=0.02)

            Xw_s = self.in_online_std.transform(X_warm)
            Yw_s = self.tg_online_std.transform(Y_warm)

            # --- Warm-start partial_fit
            self.esns = create_diverse_esns(n_models=self.n_models, base_units=25, seed=42, rls_forgetting=0.99)
            for esn in self.esns:
                esn.partial_fit(Xw_s, Yw_s)

            # per-ESN running error buffers and boost counters
            self.err_deques = [deque(maxlen=8) for _ in self.esns]
            self.boost_counters = [0 for _ in self.esns]

            self.have_warm = True
            self.get_logger().info('ESNs warm-started.')

        # keep scalers adapting
        if self.in_online_std is not None:
            self.in_online_std.update(pt.reshape(1, -1))
            if len(self.history) >= 2:
                delta = (self.history[-1] - self.history[-2]).reshape(1, -1)
                self.tg_online_std.update(delta)

    def predict_multi_esn(self):
        """Return averaged future positions (H x 2) using current window.
        Uses z-gate + Savitzky–Golay smoothed slice if available."""
        if not (self.have_warm and len(self.warmup_buffer) >= 2):
            return None

        # --- sudden-change reset (like original) ---
        if len(self.history) > 1:
            last_delta = np.linalg.norm(self.history[-1] - self.history[-2])
            if last_delta > self.sudden_change_thresh:
                for esn in self.esns:
                    # be defensive about API shape
                    try:
                        if hasattr(esn, "res") and hasattr(esn.res, "reset_state"):
                            esn.res.reset_state()
                        elif hasattr(esn, "reset"):
                            esn.reset()  # fallback if available
                    except Exception:
                        pass

        # Prepare recent (standardized) input window
        # --- Use smoothed recent slice if present; otherwise fall back to raw window ---
        if self.X_recent_for_esn is not None and len(self.X_recent_for_esn) >= 1:
            X_recent = np.array(self.X_recent_for_esn, dtype=float)
        else:
            adapt_window = min(self.adapt_window, len(self.warmup_buffer))
            X_recent = np.array(list(self.warmup_buffer)[-adapt_window:], dtype=float)

        if X_recent.shape[0] < 1:
            return None

        X_recent_s = self.in_online_std.transform(X_recent)


        # --- ESN predictions ---
        all_esn_preds = []
        for i, esn in enumerate(self.esns):
            # ---------- multi-step rollout ----------
            last_input = X_recent_s.copy()
            future_preds = []
            for _ in range(self.future_horizon):
                try:
                    delta_s = esn.run(last_input)[-1]  # standardized delta (1,2)
                except Exception:
                    delta_s = np.zeros((1, 2))

                # back to absolute pos
                delta = self.tg_online_std.inverse_transform(delta_s.reshape(1, -1))[0]
                last_pos_abs = self.in_online_std.inverse_transform(last_input[-1].reshape(1, -1))[0]
                next_pos_abs = last_pos_abs + delta
                future_preds.append(next_pos_abs)

                # roll the input window forward with the predicted point
                last_input = np.roll(last_input, -1, axis=0)
                last_input[-1] = self.in_online_std.transform(next_pos_abs.reshape(1, -1))[0]

            all_esn_preds.append(future_preds)

            # ---------- online adaptation ----------
            if len(X_recent_s) > 1:
                fit_len = min(self.delta_window, len(X_recent_s) - 1)
                if fit_len >= 1:
                    adapt_X = X_recent_s[-fit_len-1:-1]                     # (fit_len, 2)
                    adapt_Y = np.diff(X_recent_s[-fit_len-1:], axis=0)      # (fit_len, 2)

                    # predict the last standardized delta over this short window
                    try:
                        pred_last_s = esn.run(adapt_X)[-1]                   # (1,2)
                    except Exception:
                        pred_last_s = np.zeros((1, adapt_Y.shape[1]))

                    real_last = adapt_Y[-1].reshape(1, -1)                   # (1,2)
                    instantaneous_err = float(np.linalg.norm(real_last - pred_last_s))

                    # update running error + decide damping (boost vs nominal)
                    if self.err_deques is not None:
                        self.err_deques[i].append(instantaneous_err)
                        running_err = float(np.mean(self.err_deques[i]))
                    else:
                        running_err = instantaneous_err

                    if self.boost_counters is not None and self.boost_counters[i] > 0:
                        damping = self.adapt_damping_boost
                        self.boost_counters[i] -= 1
                    elif running_err > self.boost_error_thresh:
                        damping = self.adapt_damping_boost
                        if self.boost_counters is not None:
                            self.boost_counters[i] = self.boost_frames - 1
                    else:
                        damping = self.adapt_damping_nominal

                    # damp + clip the target deltas before partial_fit
                    adapt_Y_adj = np.clip(adapt_Y * damping, -self.err_clip, self.err_clip)

                    try:
                        esn.partial_fit(adapt_X, adapt_Y_adj)
                    except Exception:
                        pass

                    # clip readout weights and reservoir state (defensive checks)
                    try:
                        if hasattr(esn, "W_out"):
                            esn.W_out = np.clip(esn.W_out, -self.wout_clip, self.wout_clip)
                    except Exception:
                        pass
                    try:
                        if hasattr(esn, "res") and hasattr(esn.res, "state"):
                            esn.res.state = np.clip(esn.res.state, -self.state_clip, self.state_clip)
                    except Exception:
                        pass

        # --- keep the target standardizer adapting to real motion ---
        if len(self.warmup_buffer) >= 2:
            last_real_delta = np.array(self.warmup_buffer[-1]) - np.array(self.warmup_buffer[-2])
            try:
                self.tg_online_std.update(last_real_delta.reshape(1, -1))
            except Exception:
                pass

        # --- return averaged ESN prediction ---
        if len(all_esn_preds) > 0:
            return np.mean(np.array(all_esn_preds), axis=0)  # (H,2)
        return None


    def leg_processing(self):
        if not self.have_warm:
            return
        pred = self.predict_multi_esn()
        if pred is None or len(pred) == 0:
            return

        now = self.get_clock().now().to_msg()
        path = Path()
        path.header = Header(stamp=now, frame_id=self.frame_id)

        # start from latest observed point for continuity (optional)
        for i, (x, y) in enumerate(pred):
            ps = PoseStamped()
            ps.header.stamp = now  # same stamp; or offset if you like
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self.pub_path_.publish(path)

def main():
    rclpy.init()
    node = LegsESNPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
