"""
sensors.py — Sensor wrapper for the autonomous vehicle.

This module owns all sensor handles and provides a single high-level
`get_observation()` call that returns everything downstream code needs:
LiDAR scan, IMU readings, GPS position, derived features (yaw, velocity,
drift angle).

Design notes:
- The same Observation object is consumed by the rule-based controller
  and by RL agents. Same interface = fair comparison.
- read() returns full sensor data (debug, plotting).
- to_rl_vector() returns a fixed-size NumPy vector (ready for RL).

Coordinate convention: Webots ENU (East-North-Up) by default since R2022.
    X = East (forward when car faces +X)
    Y = North (left of car when car faces +X)
    Z = Up
The .wbt file is ENU (car height is on Z axis).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Device names — must exactly match the device's `name` field in the .wbt,
# or its PROTO defaultName if no explicit name is set.
LIDAR_NAME = "Sick LMS 291"
GPS_NAME = "gps"
GYRO_NAME = "gyro"
CAMERA_NAME = "camera"

COMPASS_NAME = None              # e.g. "compass"
IMU_NAME = "inertial unit"       # required for proper heading
ACCEL_NAME = "accelerometer"     # required for lateral accel feature
TOUCH_NAME_FRONT = "touch sensor front"
TOUCH_NAME_LEFT = "touch sensor left"
TOUCH_NAME_RIGHT = "touch sensor right"

# Number of LiDAR rays to keep in the RL observation vector.
RL_LIDAR_BINS = 32

@dataclass
class Observation:
    """All sensor data, one timestep."""

    time: float = 0.0

    # LiDAR
    lidar_ranges: np.ndarray = field(default_factory=lambda: np.array([]))
    lidar_fov: float = 0.0
    lidar_max_range: float = 0.0

    # GPS — ENU frame, in meters. [x_east, y_north, z_up]
    gps_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gps_speed: float = 0.0
    gps_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Heading (yaw) in radians, ENU convention: 0 = facing +X (East),
    # increasing counter-clockwise (positive yaw = turning left).
    heading: float = 0.0

    # IMU (only filled if IMU is present)
    imu_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Gyro angular velocity (rad/s)
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Accelerometer (m/s^2) — only filled if accel is present
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Touch sensor flag (0/1) — only filled if touch is present
    touch: float = 0.0

    # ----- Derived features -----

    # Drift angle (radians): difference between heading and the direction
    # of motion in the horizontal plane. Positive means the car points
    # to the left of the velocity vector.
    drift_angle: float = 0.0

    # Forward speed (m/s): projection of velocity onto heading direction.
    forward_speed: float = 0.0


class VehicleSensors:
    """Sensor manager. Initialises every device, exposes a single read step."""

    def __init__(self, driver, time_step: int):
        self.driver = driver
        self.time_step = time_step

        self.lidar = self._try_enable_lidar(LIDAR_NAME)
        self.gps = self._try_enable_default(GPS_NAME)
        self.gyro = self._try_enable_default(GYRO_NAME)
        self.front_camera = self._try_enable_camera(CAMERA_NAME)

        self.compass = self._try_enable_default(COMPASS_NAME) if COMPASS_NAME else None
        self.imu = self._try_enable_default(IMU_NAME) if IMU_NAME else None
        self.accel = self._try_enable_default(ACCEL_NAME) if ACCEL_NAME else None
        self.touch_front = self._try_enable_default(TOUCH_NAME_FRONT) if TOUCH_NAME_FRONT else None
        self.touch_left = self._try_enable_default(TOUCH_NAME_LEFT) if TOUCH_NAME_LEFT else None
        self.touch_right = self._try_enable_default(TOUCH_NAME_RIGHT) if TOUCH_NAME_RIGHT else None

        if self.lidar is not None:
            self.lidar_horizontal_resolution = self.lidar.getHorizontalResolution()
            self.lidar_fov = self.lidar.getFov()
            self.lidar_max_range = self.lidar.getMaxRange()
        else:
            self.lidar_horizontal_resolution = 0
            self.lidar_fov = 0.0
            self.lidar_max_range = 0.0

        self._last_gps_position = None

    # ---------- private device helpers ----------

    def _try_enable_default(self, name: str):
        """Get a device by name and call .enable(time_step). Returns None if absent."""
        device = self.driver.getDevice(name)
        if device is None:
            print(f"[sensors] WARNING: device '{name}' not found.", flush=True)
            return None
        try:
            device.enable(self.time_step)
        except AttributeError:
            pass
        return device

    def _try_enable_lidar(self, name: str):
        device = self.driver.getDevice(name)
        if device is None:
            print(f"[sensors] WARNING: lidar '{name}' not found.", flush=True)
            return None
        device.enable(self.time_step)
        device.enablePointCloud()
        return device

    def _try_enable_camera(self, name: str):
        device = self.driver.getDevice(name)
        if device is None:
            print(f"[sensors] WARNING: camera '{name}' not found.", flush=True)
            return None
        device.enable(self.time_step)
        return device

    # ---------- public API ----------

    def read(self, current_time: float) -> Observation:
        """Read all sensors and return a fresh Observation. Call every tick."""
        obs = Observation(time=current_time)

        # --- LiDAR ---
        if self.lidar is not None:
            ranges = self.lidar.getRangeImage()
            obs.lidar_ranges = (
                np.asarray(ranges, dtype=np.float32)
                if ranges is not None and len(ranges) > 0
                else np.array([], dtype=np.float32)
            )
            obs.lidar_fov = self.lidar_fov
            obs.lidar_max_range = self.lidar_max_range

        # --- GPS ---
        # ENU world frame: position is [x_east, y_north, z_up].
        if self.gps is not None:
            pos = np.asarray(self.gps.getValues(), dtype=np.float32)
            obs.gps_position = pos

            try:
                obs.gps_speed = float(self.gps.getSpeed())
                obs.gps_velocity = np.asarray(
                    self.gps.getSpeedVector(), dtype=np.float32
                )
            except (AttributeError, TypeError):
                if self._last_gps_position is not None:
                    dt = self.time_step / 1000.0
                    vel = (pos - self._last_gps_position) / dt
                    obs.gps_velocity = vel
                    obs.gps_speed = float(np.linalg.norm(vel))
            self._last_gps_position = pos.copy()

        # --- Gyro ---
        # gyro.getValues() returns [wx, wy, wz] around world axes (ENU).
        # In ENU, wz is the yaw rate (around the vertical axis).
        if self.gyro is not None:
            obs.gyro = np.asarray(self.gyro.getValues(), dtype=np.float32)

        # --- Heading source ---
        # Priority order: IMU yaw > compass > integrated gyro fallback.
        # Without IMU/compass, we cannot get an absolute heading from a
        # single tick; we'd have to integrate gyro from a known initial
        # heading. For now, with neither IMU nor compass on the car, we
        # leave heading at 0 and warn the developer.
        if self.imu is not None:
            rpy = self.imu.getRollPitchYaw()
            obs.imu_rpy = np.asarray(rpy, dtype=np.float32)
            obs.heading = float(rpy[2])  # yaw
        elif self.compass is not None:
            north = self.compass.getValues()
            # ENU convention: north is [0, 1, 0]. Compass returns the
            # north vector expressed in the car frame, so heading is the
            # angle that takes us from car-forward to world-north.
            # In ENU, atan2(-north[0], north[1]) gives yaw in [-pi, pi].
            obs.heading = math.atan2(-float(north[0]), float(north[1]))
        # else: heading stays 0.0 — drift_angle and forward_speed below
        # will be approximate. Add an IMU to the car for proper values.

        # --- Accelerometer (optional) ---
        if self.accel is not None:
            obs.accel = np.asarray(self.accel.getValues(), dtype=np.float32)

        # --- Touch ---
        if all([touch is not None for touch in [self.touch_front, self.touch_left, self.touch_right]]):
            v_front = self.touch_front.getValue()
            v_left = self.touch_left.getValue()
            v_right = self.touch_right.getValue()
            obs.touch = float(max([v_front, v_left, v_right]))

        # --- Derived features (ENU) ---
        # Velocity in horizontal plane: use X and Y components, ignore Z.
        if obs.gps_speed > 0.1:
            vx = float(obs.gps_velocity[0])
            vy = float(obs.gps_velocity[1])
            velocity_angle = math.atan2(vy, vx)
            obs.drift_angle = self._wrap_angle(obs.heading - velocity_angle)

            heading_unit = np.array(
                [math.cos(obs.heading), math.sin(obs.heading), 0.0],
                dtype=np.float32,
            )
            obs.forward_speed = float(np.dot(obs.gps_velocity, heading_unit))
        else:
            obs.drift_angle = 0.0
            obs.forward_speed = 0.0

        return obs

    # ---------- RL observation builder ----------

    def to_rl_vector(self, obs: Observation) -> np.ndarray:
        """Flat vector for an RL network.

        Layout:
            [0:RL_LIDAR_BINS]   - downsampled & normalised lidar
            [RL_LIDAR_BINS]     - forward_speed (normalised)
            [RL_LIDAR_BINS+1]   - drift_angle (radians)
            [RL_LIDAR_BINS+2]   - yaw rate (gyro Z, rad/s)
            [RL_LIDAR_BINS+3]   - lateral accel (m/s^2, 0 if no accel sensor)
            [RL_LIDAR_BINS+4]   - touch flag (0/1, 0 if no touch sensor)

        Total length: RL_LIDAR_BINS + 5
        """
        lidar_bins = self._downsample_lidar(obs.lidar_ranges, RL_LIDAR_BINS)
        if obs.lidar_max_range > 0:
            lidar_bins = lidar_bins / obs.lidar_max_range
        lidar_bins = np.clip(lidar_bins, 0.0, 1.0)

        norm_speed = float(obs.forward_speed) / 40.0

        # In ENU, yaw rate is gyro Z (index 2).
        yaw_rate = float(obs.gyro[2]) if obs.gyro.size >= 3 else 0.0

        # Lateral accel: in vehicle frame this would be Y. World-frame
        # approximation is fine while the car has small yaw.
        lat_accel = float(obs.accel[1]) if obs.accel.size >= 2 else 0.0

        extras = np.array(
            [norm_speed, obs.drift_angle, yaw_rate, lat_accel, obs.touch],
            dtype=np.float32,
        )

        return np.concatenate([lidar_bins.astype(np.float32), extras])

    # ---------- helpers ----------

    @staticmethod
    def _downsample_lidar(ranges: np.ndarray, n_bins: int) -> np.ndarray:
        if ranges.size == 0 or n_bins <= 0:
            return np.zeros(n_bins, dtype=np.float32)

        clean = np.where(np.isfinite(ranges), ranges, np.inf)
        finite = clean[np.isfinite(clean)]
        max_finite = float(np.max(finite)) if finite.size > 0 else 0.0
        clean = np.where(np.isinf(clean), max_finite, clean)

        chunks = np.array_split(clean, n_bins)
        return np.array([float(np.mean(c)) for c in chunks], dtype=np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi