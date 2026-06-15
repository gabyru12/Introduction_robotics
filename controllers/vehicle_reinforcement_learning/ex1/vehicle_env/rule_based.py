"""
rule_based.py — Rule-based baseline controller + oval track follower.

Two classes are defined here:

RuleBasedAgent — the paper's "no learning" baseline.  Uses only LiDAR
    L/R imbalance to steer.  Deliberately simple so any RL gain is real.

OvalRaceController — Stanley controller that follows the oval centreline
    defined in oval_track.wbt.  Uses GPS + IMU as primary steering with
    LiDAR only as an emergency-brake safety layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sensors import Observation

try:
    from controller import Keyboard as _Keyboard
except ImportError:
    _Keyboard = None  # type: ignore[assignment]


@dataclass
class RuleBasedConfig:
    """Tunable parameters. Keep these in one place so we can sweep them."""

    # Distance under which we treat an obstacle as critical (meters).
    emergency_distance: float = 5.0

    # Distance at which we start slowing down (meters).
    slowdown_distance: float = 15.0

    # Max steering angle output (in [-1, 1] normalised units).
    max_steering: float = 1.0

    # Steering gain on the lidar L/R imbalance signal.
    steering_gain: float = 1.5

    # Front window: number of rays on each side of centre to consider
    # for "what's straight ahead". Tune to the LiDAR's resolution and FOV.
    front_window_half: int = 30

    # Throttle bounds.
    min_throttle: float = 0.0
    max_throttle: float = 0.6   # capped on purpose — baseline is meant to be safe

    # Cornering speed reduction: when |steering| is large, ease the throttle.
    cornering_speed_factor: float = 0.5


class RuleBasedAgent:
    """Hand-crafted controller. Same interface as future RL agents."""

    def __init__(self, config: RuleBasedConfig | None = None):
        self.cfg = config or RuleBasedConfig()

    def reset(self) -> None:
        """Called when episode resets. Nothing to reset for stateless rules."""
        pass

    def act(self, obs: Observation) -> tuple[float, float, float]:
        """Return (steering, throttle, brake), all in normalised ranges.

        steering in [-1, 1]
        throttle in [0, 1]
        brake    in [0, 1]
        """
        ranges = obs.lidar_ranges

        # Sanity: no lidar data → coast.
        if ranges.size == 0:
            return 0.0, 0.2, 0.0

        n = ranges.size
        centre = n // 2
        half = min(self.cfg.front_window_half, centre)

        # Replace inf/nan with max range so arithmetic works.
        clean = np.where(
            np.isfinite(ranges),
            ranges,
            obs.lidar_max_range if obs.lidar_max_range > 0 else 80.0,
        )

        front = clean[centre - half : centre + half]
        min_front = float(np.min(front)) if front.size > 0 else float("inf")

        # Emergency brake.
        if min_front < self.cfg.emergency_distance:
            # Steer toward whichever side has more average space.
            left_mean = float(np.mean(clean[: centre]))
            right_mean = float(np.mean(clean[centre:]))
            steer = self.cfg.max_steering if left_mean > right_mean else -self.cfg.max_steering
            return steer, 0.0, 1.0

        # Normal driving: steering from left/right imbalance in the front
        # window. Positive imbalance (more space on right) → steer right.
        left_front = float(np.mean(clean[centre - half : centre]))
        right_front = float(np.mean(clean[centre : centre + half]))
        imbalance = right_front - left_front
        norm_factor = max(left_front + right_front, 1e-3)
        steer_raw = self.cfg.steering_gain * imbalance / norm_factor
        steer = float(np.clip(steer_raw, -self.cfg.max_steering, self.cfg.max_steering))

        # Throttle: more space ahead = more gas. Linearly ramp up between
        # emergency_distance and slowdown_distance.
        if min_front >= self.cfg.slowdown_distance:
            throttle = self.cfg.max_throttle
        else:
            span = self.cfg.slowdown_distance - self.cfg.emergency_distance
            t = (min_front - self.cfg.emergency_distance) / max(span, 1e-3)
            throttle = self.cfg.min_throttle + t * (
                self.cfg.max_throttle - self.cfg.min_throttle
            )

        # Ease off the throttle in tight corners.
        throttle *= 1.0 - self.cfg.cornering_speed_factor * abs(steer)
        throttle = float(np.clip(throttle, 0.0, 1.0))

        return steer, throttle, 0.0


# ---------------------------------------------------------------------------
# Oval track follower (Stanley controller)
# ---------------------------------------------------------------------------

# Maximum physical steering angle of the BmwX5 PROTO (radians).
# Mirrors the constant in my_controller.py — kept here so this module is
# self-contained when unit-tested outside Webots.
_MAX_STEER_RAD = 0.75

# Target cruising speed used to normalise throttle output (km/h).
# Must match CRUISING_SPEED_KMH in my_controller.py.
_CRUISING_SPEED_KMH = 60.0


@dataclass
class OvalRaceConfig:
    """Geometry and tuning parameters for the oval track controller."""

    # ---- Track geometry (from oval_track.wbt) ----
    straight_top_y: float = 25.0        # Y coordinate of the top straight centreline
    straight_bot_y: float = -25.0       # Y coordinate of the bottom straight centreline
    left_curve_cx: float = -30.0        # X of left semicircle centre
    right_curve_cx: float = 30.0        # X of right semicircle centre
    curve_cy: float = 0.0               # Y of both curve centres
    curve_radius: float = 25.0          # Centreline radius of each semicircle
    straight_x_min: float = -30.0       # X limits of the straight sections
    straight_x_max: float = 30.0

    # ---- Speed targets ----
    straight_speed_kmh: float = 50.0    # Target speed on the straights
    curve_speed_kmh: float = 28.0       # Target speed inside the curves

    # ---- Stanley controller ----
    stanley_k: float = 0.8              # Cross-track error gain
    v_min: float = 1.0                  # Minimum speed used in atan2 (avoids div/0)

    # ---- LiDAR safety layer ----
    emergency_dist: float = 4.0         # If min front LiDAR < this, brake (meters)
    front_window_half: int = 20         # Half-width of the front LiDAR window (rays)

    # ---- Yaw-rate speed reduction ----
    yaw_rate_threshold: float = 0.25    # rad/s above which we trim speed
    yaw_rate_gain: float = 0.6         # Speed multiplier applied per unit of excess yaw rate


class OvalRaceController:
    """Stanley path-following controller for the oval track.

    Primary steering: GPS position + IMU heading → Stanley formula.
    Speed control:    segment-type (straight vs curve) + yaw-rate feedback.
    Safety layer:     LiDAR emergency brake (same logic as RuleBasedAgent).
    """

    def __init__(self, config: OvalRaceConfig | None = None) -> None:
        self.cfg = config or OvalRaceConfig()

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(self, obs: Observation) -> tuple[float, float, float]:
        """Return (steering, throttle, brake) in normalised ranges."""

        # --- Safety layer first ---
        emergency = self._lidar_emergency(obs)
        if emergency is not None:
            return emergency

        # --- GPS/IMU availability check ---
        px = float(obs.gps_position[0])
        py = float(obs.gps_position[1])
        heading = obs.heading  # radians, ENU convention

        # --- Find nearest point on centreline ---
        cx, cy, path_heading, _ = self._nearest_on_centreline(px, py)

        # --- Cross-track error (signed: positive = car is left of path) ---
        # Path normal points 90° CCW from tangent, i.e., to the left of travel.
        perp_x = -math.sin(path_heading)
        perp_y = math.cos(path_heading)
        cte = (px - cx) * perp_x + (py - cy) * perp_y

        # --- Heading error ---
        psi_e = _wrap_angle(heading - path_heading)

        # --- Stanley steering ---
        speed_ms = max(abs(obs.forward_speed), self.cfg.v_min)
        delta = psi_e + math.atan2(self.cfg.stanley_k * cte, speed_ms)
        steer = float(np.clip(delta / _MAX_STEER_RAD, -1.0, 1.0))

        # --- Speed control ---
        throttle = self._speed_control(obs, px, py)

        return steer, throttle, 0.0

    # ------------------------------------------------------------------
    # Centreline geometry
    # ------------------------------------------------------------------

    def _nearest_on_centreline(
        self, px: float, py: float
    ) -> tuple[float, float, float, float]:
        """Return (cx, cy, path_heading, distance) for the nearest point."""
        c = self.cfg
        candidates = [
            self._nearest_on_top_straight(px, py),
            self._nearest_on_bot_straight(px, py),
            self._nearest_on_arc(
                px, py,
                c.left_curve_cx, c.curve_cy,
                math.pi / 2.0, 3.0 * math.pi / 2.0,   # θ ∈ [90°, 270°]
            ),
            self._nearest_on_arc(
                px, py,
                c.right_curve_cx, c.curve_cy,
                -math.pi / 2.0, math.pi / 2.0,          # θ ∈ [−90°, 90°]
            ),
        ]
        return min(candidates, key=lambda tup: tup[3])

    def _nearest_on_top_straight(
        self, px: float, py: float
    ) -> tuple[float, float, float, float]:
        c = self.cfg
        cx = float(np.clip(px, c.straight_x_min, c.straight_x_max))
        cy = c.straight_top_y
        d = math.hypot(px - cx, py - cy)
        return cx, cy, math.pi, d   # heading West

    def _nearest_on_bot_straight(
        self, px: float, py: float
    ) -> tuple[float, float, float, float]:
        c = self.cfg
        cx = float(np.clip(px, c.straight_x_min, c.straight_x_max))
        cy = c.straight_bot_y
        d = math.hypot(px - cx, py - cy)
        return cx, cy, 0.0, d       # heading East

    def _nearest_on_arc(
        self,
        px: float, py: float,
        arc_cx: float, arc_cy: float,
        theta_min: float, theta_max: float,
    ) -> tuple[float, float, float, float]:
        """Nearest point on a CCW arc between theta_min and theta_max."""
        c = self.cfg
        rel_x = px - arc_cx
        rel_y = py - arc_cy
        theta = math.atan2(rel_y, rel_x)
        theta = float(np.clip(theta, theta_min, theta_max))
        nx = arc_cx + c.curve_radius * math.cos(theta)
        ny = arc_cy + c.curve_radius * math.sin(theta)
        d = math.hypot(px - nx, py - ny)
        # Tangent of CCW arc: perpendicular to radius, rotated 90° CCW.
        # heading = atan2(cos θ, −sin θ)
        tangent = math.atan2(math.cos(theta), -math.sin(theta))
        return nx, ny, tangent, d

    # ------------------------------------------------------------------
    # Speed control
    # ------------------------------------------------------------------

    def _speed_control(self, obs: Observation, px: float, py: float) -> float:
        c = self.cfg
        d_left = math.hypot(px - c.left_curve_cx, py - c.curve_cy)
        d_right = math.hypot(px - c.right_curve_cx, py - c.curve_cy)
        on_curve = d_left < c.curve_radius + 6.0 or d_right < c.curve_radius + 6.0

        target_kmh = c.curve_speed_kmh if on_curve else c.straight_speed_kmh

        yaw_rate = abs(float(obs.gyro[2])) if obs.gyro.size >= 3 else 0.0
        excess = max(0.0, yaw_rate - c.yaw_rate_threshold)
        if excess > 0.0:
            target_kmh *= max(0.4, 1.0 - c.yaw_rate_gain * excess)

        throttle = target_kmh / _CRUISING_SPEED_KMH
        return float(np.clip(throttle, 0.0, 1.0))

    # ------------------------------------------------------------------
    # LiDAR safety layer
    # ------------------------------------------------------------------

    def _lidar_emergency(
        self, obs: Observation
    ) -> tuple[float, float, float] | None:
        """Return an emergency action if the LiDAR detects imminent collision."""
        ranges = obs.lidar_ranges
        if ranges.size == 0:
            return None
        c = self.cfg
        n = ranges.size
        centre = n // 2
        half = min(c.front_window_half, centre)
        max_r = obs.lidar_max_range if obs.lidar_max_range > 0 else 80.0
        clean = np.where(np.isfinite(ranges), ranges, max_r)
        front = clean[centre - half : centre + half]
        if front.size == 0 or float(np.min(front)) >= c.emergency_dist:
            return None
        left_mean = float(np.mean(clean[:centre]))
        right_mean = float(np.mean(clean[centre:]))
        steer = 1.0 if left_mean > right_mean else -1.0
        return steer, 0.0, 1.0


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Keyboard controller
# ---------------------------------------------------------------------------

# Steering ramp: 0.04 = 1/25 → zero to full lock in 25 steps × 16 ms = 400 ms.
KB_STEER_INCREMENT: float = 0.04
# Return-to-centre: 0.0625 = 1/16 → full lock to centre in 16 steps × 16 ms = 256 ms.
KB_STEER_CENTER_RATE: float = 0.0625
KB_THROTTLE_PRESSED: float = 1.5
KB_BRAKE_INTENSITY: float = 0.6
KB_HANDBRAKE_INTENSITY: float = 0.9
KB_ENGINE_BRAKE_DEFAULT: float = 0.05


@dataclass
class KeyboardConfig:
    """Tunable parameters for KeyboardController."""

    steer_increment: float = KB_STEER_INCREMENT
    steer_center_rate: float = KB_STEER_CENTER_RATE
    throttle_pressed: float = KB_THROTTLE_PRESSED
    brake_intensity: float = KB_BRAKE_INTENSITY
    handbrake_intensity: float = KB_HANDBRAKE_INTENSITY
    engine_brake: float = KB_ENGINE_BRAKE_DEFAULT


class KeyboardController:
    """Manual driving via keyboard. Arrows or WASD.

    UP/W=throttle, DOWN/S=brake, LEFT/A=steer left, RIGHT/D=steer right.
    Simultaneous keys are supported (e.g. UP+LEFT = throttle + steer left).
    Brake overrides throttle. Steering ramps smoothly; throttle/brake instant.
    """

    def __init__(self, keyboard, config: KeyboardConfig | None = None) -> None:
        if _Keyboard is None:
            raise RuntimeError(
                "KeyboardController requires the Webots 'controller' package."
            )
        self._kb = keyboard
        self.cfg = config or KeyboardConfig()
        self._steer: float = 0.0
        self._keys_forward   = frozenset({_Keyboard.UP,   ord('W')})
        self._keys_back      = frozenset({_Keyboard.DOWN,  ord('S')})
        self._keys_left      = frozenset({_Keyboard.LEFT,  ord('A')})
        self._keys_right     = frozenset({_Keyboard.RIGHT, ord('D')})
        self._keys_handbrake = frozenset({ord(' ')})

    def reset(self) -> None:
        """Reset steering to centre when the simulation resets."""
        self._steer = 0.0

    def _read_pressed(self) -> set[int]:
        """Drain the Webots key buffer; return all currently-pressed key codes."""
        pressed: set[int] = set()
        k = self._kb.getKey()
        while k != -1:
            pressed.add(k)
            k = self._kb.getKey()
        return pressed

    def act(self, driver, obs: Observation) -> tuple[float, float, float]:
        """Return (steering, throttle, brake) from the current keyboard state.

        Args:
            obs: Sensor observation (unused — control is driven by hardware input).

        Returns:
            Normalised (steering ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]).
        """
        # obs unused — keyboard reads hardware input directly
        pressed = self._read_pressed()
        cfg = self.cfg

        want_left  = bool(pressed & self._keys_left)
        want_right = bool(pressed & self._keys_right)

        if want_left and not want_right:
            #car_node = driver.getFromDef("VEHICLE")
            #vel = car_node.getVelocity()
            #car_node.setVelocity([
            #    vel[0], vel[1], vel[2],
            #    vel[3], vel[4], 0.0
            #])
            #self._steer = min(1.0, self._steer + cfg.steer_increment)
            self._steer = max(-1.0, self._steer - cfg.steer_increment)
        elif want_right and not want_left:
            self._steer = min(1.0, self._steer + cfg.steer_increment)
        else:
            if self._steer > 0.0:
                self._steer = max(0.0, self._steer - cfg.steer_center_rate)
            elif self._steer < 0.0:
                self._steer = min(0.0, self._steer + cfg.steer_center_rate)

        # Throttle / brake (handbrake > regular brake > throttle > coast)
        want_fwd       = bool(pressed & self._keys_forward)
        want_back      = bool(pressed & self._keys_back)
        want_handbrake = bool(pressed & self._keys_handbrake)
        if want_handbrake:
            return self._steer, 0.0, cfg.handbrake_intensity
        if want_back:
            return self._steer, 0.0, cfg.brake_intensity
        if want_fwd:
            return self._steer, cfg.throttle_pressed, 0.0

        return self._steer, 0.0, cfg.engine_brake


# ---------------------------------------------------------------------------
# Drift demo controller (steady-state donut)
# ---------------------------------------------------------------------------
#
# This controller is designed to perform continuous counter-clockwise donuts
# on an open surface (no obstacles). Designed for demo recording.
#
# State machine:
#   ACCELERATE - drive straight until target speed is reached
#   INITIATE   - hard left steer + brief brake to break rear traction
#   DONUT      - hold steering + partial throttle to sustain steady-state slide
#
# There is no exit state - once in DONUT the car keeps drifting indefinitely.


@dataclass
class DriftDemoConfig:
    """Steady-state donut controller. Counter-clockwise (turn left).

    The car accelerates in a straight line, then initiates a left-hand donut
    and maintains it. No path following - this is a stationary spinning demo.
    """

    # ---- Acceleration phase ----
    target_speed_ms: float = 15.0        # m/s before initiating the donut (54 km/h)

    # ---- Initiation phase (break rear traction) ----
    initiate_steer: float = 1.0          # full left lock
    initiate_throttle: float = 0.0
    initiate_brake: float = 0.6          # brake to unload rear and rotate
    initiate_duration_s: float = 0.4     # duration of the trigger pulse

    # ---- Donut phase (steady-state) ----
    # The car holds a near-constant left-turn radius while sliding.
    donut_base_steer: float = 0.85       # base left lock during donut
    donut_throttle: float = 0.55         # partial throttle keeps rear spinning
    donut_steer_drift_gain: float = 0.4  # how much to back off steering as drift grows

    # ---- Drift angle target ----
    # Positive drift (car points left of velocity) is what we want for a
    # left-hand donut. We aim to keep drift in this range.
    drift_target: float = 0.6            # rad (~34 degrees), the "sweet spot"


class DriftDemoController:
    """Steady-state donut controller for demo recording.

    Use POLICY = "drift_demo" in my_controller.py.
    Designed to run on an open surface with no obstacles.
    """

    ACCELERATE = "ACCELERATE"
    INITIATE   = "INITIATE"
    DONUT      = "DONUT"

    def __init__(self, config: DriftDemoConfig | None = None) -> None:
        self.cfg = config or DriftDemoConfig()
        self.state = self.ACCELERATE
        self._state_entry_time: float = 0.0

    def reset(self) -> None:
        self.state = self.ACCELERATE
        self._state_entry_time = 0.0

    def _switch(self, new_state: str, now: float) -> None:
        print(f"[donut] {self.state} -> {new_state} at t={now:.2f}", flush=True)
        self.state = new_state
        self._state_entry_time = now

    def act(self, obs: Observation) -> tuple[float, float, float]:
        c = self.cfg
        now = obs.time
        speed = abs(obs.forward_speed)

        # ---------- ACCELERATE ----------
        if self.state == self.ACCELERATE:
            if speed >= c.target_speed_ms:
                self._switch(self.INITIATE, now)
            else:
                # Drive straight, throttle proportional to speed deficit
                speed_error = c.target_speed_ms - speed
                throttle = float(np.clip(speed_error / 5.0, 0.3, 1.0))
                return 0.0, throttle, 0.0

        # ---------- INITIATE ----------
        if self.state == self.INITIATE:
            if (now - self._state_entry_time) >= c.initiate_duration_s:
                self._switch(self.DONUT, now)
            else:
                return c.initiate_steer, c.initiate_throttle, c.initiate_brake

        # ---------- DONUT (steady-state) ----------
        if self.state == self.DONUT:
            drift = obs.drift_angle  # positive = car points left of velocity

            # Steering control: hold base steer, reduce as drift grows past target.
            # If drift is bigger than target, ease the steering to widen the circle
            # and let drift come back. If smaller, hold full base steer.
            drift_excess = max(0.0, drift - c.drift_target)
            steer = c.donut_base_steer - c.donut_steer_drift_gain * drift_excess
            steer = float(np.clip(steer, 0.3, 1.0))

            return steer, c.donut_throttle, 0.0

        # Should never reach here
        return 0.0, 0.0, 0.0