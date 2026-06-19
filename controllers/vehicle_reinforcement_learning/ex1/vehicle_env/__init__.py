import time
import os

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import math

from vehicle import Driver

import json


# TODO: (RL Research)
# - how to discretize lidar rays into bins, is it better for reinforcement learning then keeping all of them
# - how to set reward values in order to not overprioritize a certain behaviour
# - how much does adding another action space variable affect the training time of the RL model (adding vs removing the brake variable)
# - is there any study on changing environments physics to aid in training on unstable environment


# =========================================================
# Track Functions
# =========================================================

def line_points(x0, y0, x1, y1, n=100):
    x = np.linspace(x0, x1, n)
    y = np.linspace(y0, y1, n)
    return np.column_stack((x, y))


def arc_points(cx, cy, radius, start_angle, end_angle, n=100):
    theta = np.linspace(start_angle, end_angle, n)

    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    return np.column_stack((x, y))


def get_centerline(world_name):
    center_x = 150
    center_y = 0
    centerline = None
    if world_name == "village_winter_track.wbt":
        seg1 = line_points(135, -25,180, -25,n=50)
        seg2 = arc_points(cx=180,cy=0,radius=25,start_angle=-np.pi/2,end_angle=np.pi/2,n=100)
        seg3 = line_points(180, 25,120, 25,n=100)
        seg4 = arc_points(cx=120,cy=0,radius=25,start_angle=np.pi/2,end_angle=3*np.pi/2,n=100)
        seg5 = line_points(120, -25,135, -25,n=50)
        centerline = np.vstack([seg1[:-1],seg2[:-1],seg3[:-1],seg4[:-1],seg5])
    elif world_name == "village_winter_circle_arena_35.wbt":
        theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        centerline = np.column_stack([center_x + 25 * np.cos(theta), 25 * np.sin(theta)])
    elif world_name == "village_winter_circle_arena_40.wbt":
        theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        centerline = np.column_stack([center_x + 30 * np.cos(theta), 25 * np.sin(theta)])
    elif world_name == "village_winter_circle_arena_45.wbt":
        theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        centerline = np.column_stack([center_x + 35 * np.cos(theta), 35 * np.sin(theta)])
    elif world_name == "village_winter_circle_arena_50.wbt":
        theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        centerline = np.column_stack([center_x + 40 * np.cos(theta), 40 * np.sin(theta)])

    return centerline


def closest_centerline_point(car_pos, centerline):
    car_pos = np.asarray(car_pos[:2])

    diff = centerline - car_pos
    dist2 = np.sum(diff**2, axis=1)

    idx = np.argmin(dist2)

    return idx, centerline[idx], math.sqrt(dist2[idx])


def compute_centerline_s(centerline):
    diffs = np.diff(centerline, axis=0)

    seg_lengths = np.linalg.norm(diffs, axis=1)

    s = np.concatenate([[0], np.cumsum(seg_lengths)])

    return s


def facilitate_driving(driver, previous_easier_control, world_name):
    car_node = driver.getFromDef("VEHICLE")
    translation_field = car_node.getField("translation")
    x = translation_field.getSFVec3f()[0]
    just_turned_easier_control_on = False
    just_turned_easier_control_off = False
    if "arena" in world_name:
        easier_control = True
    else:
        if x > 170 or x < 130:
            if not previous_easier_control:
                just_turned_easier_control_on = True
            easier_control = True
        else:
            if previous_easier_control:
                just_turned_easier_control_off = True
            easier_control = False

    vel = car_node.getVelocity()
    if easier_control:
        increase_rate = 0
        if driver.getSteeringAngle() < -0.2:
            if vel[5] > 0:
                increase_rate = 0.02
            else:
                increase_rate = 0.04
        elif driver.getSteeringAngle() > 0.2:
            if vel[5] < 0:
                increase_rate = -0.02
            else:
                increase_rate = -0.04

        car_node.setVelocity([
            vel[0], vel[1], vel[2],
            vel[3], vel[4], vel[5] + increase_rate
        ])
    else:
        if just_turned_easier_control_off:
            car_node.setVelocity([
                vel[0], vel[1], vel[2],
                vel[3], vel[4], min(max(vel[5], -0.2), 0.2)
            ])
            #print(f"position: {x} | new_angular_velocity_z: {min(max(vel[5], -0.5), 0.5)}")

    return easier_control


# =========================================================
# Utility Functions
# =========================================================

def normalize_signed(value, max_abs):
    return np.clip(value / max_abs, -1.0, 1.0)


def normalize_unsigned(value, max_value):
    return np.clip(value / max_value, 0.0, 1.0)


def _downsample_lidar(ranges: np.ndarray, n_bins: int) -> np.ndarray:
    if ranges.size == 0 or n_bins <= 0:
        return np.zeros(n_bins, dtype=np.float32)

    clean = np.where(np.isfinite(ranges), ranges, np.inf)

    finite = clean[np.isfinite(clean)]
    max_finite = float(np.max(finite)) if finite.size > 0 else 0.0

    clean = np.where(np.isinf(clean), max_finite, clean)

    chunks = np.array_split(clean, n_bins)

    return np.array(
        [float(np.mean(c)) for c in chunks],
        dtype=np.float32
    )


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# =========================================================
# Observations Function
# =========================================================

def build_observation(
    driver,
    centerline,
    lidar,
    gyro,
    gps,
    compass,
    prev_action,
    n_lidar_bins=12,
    MAX_LIDAR_RANGE=30.0,
    MAX_CRUISING_SPEED=250.0,
    MAX_LONGITUDINAL_VELOCITY=30.0,
    MAX_LATERAL_VELOCITY=30.0,
    MAX_LINEAR_VELOCITY=30.0,
    MAX_YAW_RATE=1,
    MAX_STEER_ANGLE=1.0,
    MAX_CAR_HEADING_ANGLE=180,
    MAX_ROAD_HEADING_ANGLE=180,
    MAX_DRIFT_ANGLE=90,
    MAX_DIS2CENTERLINE=8
):
    # -----------------------------------------------------
    # LIDAR (NECESSARY)
    # -----------------------------------------------------
    lidar_values = lidar.getRangeImage()
    lidar_values = (
        np.asarray(lidar_values, dtype=np.float32)
        if lidar_values is not None and len(lidar_values) > 0
        else np.array([], dtype=np.float32)
    )
    lidar_bins = _downsample_lidar(lidar_values, n_lidar_bins)
    # TODO: is it better for distances closer to the wall be close to 1 or should we maintain as it is meaning
    #  normalised values are 0 if they are closer to the wall and 1 whenever they are as far as possible
    lidar_norm = np.clip(lidar_bins / MAX_LIDAR_RANGE,0.0,1.0)

    # -----------------------------------------------------
    # VEHICLE STATE (NECESSARY)
    # -----------------------------------------------------
    cruising_speed = driver.getCurrentSpeed()
    cruising_speed_norm = normalize_unsigned(abs(cruising_speed), MAX_CRUISING_SPEED)
    throttle = driver.getThrottle()
    steering_angle = driver.getSteeringAngle()
    steering_norm = normalize_signed(steering_angle, MAX_STEER_ANGLE)

    # -----------------------------------------------------
    # GPS
    # -----------------------------------------------------
    gps_velocity_vector = np.asarray(gps.getSpeedVector(), dtype=np.float32)

    # -----------------------------------------------------
    # LONGITUDINAL VELOCITY & LATERAL VELOCITY & LINEAR VELOCITY
    # -----------------------------------------------------
    north = compass.getValues()
    heading = math.atan2(-float(north[0]), float(north[1]))
    heading_unit = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=np.float32)
    right_unit = np.array([-math.sin(heading), math.cos(heading), 0.0], dtype=np.float32)
    longitudinal_velocity = float(np.dot(gps_velocity_vector, heading_unit))
    longitudinal_velocity_norm = normalize_unsigned(longitudinal_velocity, MAX_LONGITUDINAL_VELOCITY)
    lateral_velocity = float(np.dot(gps_velocity_vector, right_unit))
    lateral_velocity_norm = normalize_unsigned(lateral_velocity, MAX_LATERAL_VELOCITY)
    linear_velocity = math.hypot(longitudinal_velocity, lateral_velocity)
    linear_velocity_norm = normalize_unsigned(linear_velocity, MAX_LINEAR_VELOCITY)

    # -----------------------------------------------------
    # DISTANCE TO CENTERLINE & DRIFT ANGLE & SLIP ANGLE
    # -----------------------------------------------------
    car_pos = driver.getFromDef("VEHICLE").getField("translation").getSFVec3f()
    idx, centerline_point, dist2_centerline_point = closest_centerline_point(car_pos[:2], centerline)
    road_heading = centerline[(idx + 1) % len(centerline)] - centerline[idx]
    road_heading = math.atan2(road_heading[1], road_heading[0])
    road_heading_unit = np.array([round(math.cos(road_heading),3), round(math.sin(road_heading),3), 0.0], dtype=np.float32)

    north = compass.getValues()
    car_heading_rad = -math.atan2(-float(north[0]), float(north[1]))
    car_heading_unit = np.array([round(math.cos(car_heading_rad),3), round(math.sin(car_heading_rad),3), 0.0], dtype=np.float32)

    velocity_heading_rad = math.atan2(gps_velocity_vector[1], gps_velocity_vector[0])
    slip_angle_rad = _wrap_angle(car_heading_rad - velocity_heading_rad)

    heading_error_deg = int(_wrap_angle(car_heading_rad - road_heading) * (180 / math.pi))
    dist2centerline_point = dist2_centerline_point
    car_heading_deg = int(car_heading_rad * 180 / math.pi)
    road_heading_deg = int(road_heading * 180 / math.pi)
    velocity_heading_deg = int(velocity_heading_rad * 180 / math.pi)
    slip_angle_deg = int(slip_angle_rad * 180 / math.pi)

    car_heading_norm = normalize_signed(car_heading_deg, MAX_CAR_HEADING_ANGLE)
    road_heading_norm = normalize_signed(road_heading_deg, MAX_ROAD_HEADING_ANGLE)
    velocity_heading_norm = normalize_signed(velocity_heading_deg, MAX_ROAD_HEADING_ANGLE)
    heading_error_deg_norm = normalize_signed(heading_error_deg, MAX_DRIFT_ANGLE)
    slip_angle_deg_norm = normalize_signed(slip_angle_deg, MAX_DRIFT_ANGLE)
    dist2centerline_point_norm = normalize_unsigned(dist2_centerline_point, MAX_DIS2CENTERLINE)

    # -----------------------------------------------------
    # YAW RATE (OPTIONAL | UNDERSTAND)
    # -----------------------------------------------------
    gyro_values = gyro.getValues()
    yaw_rate = gyro_values[2]
    yaw_rate_norm = normalize_signed(yaw_rate, MAX_YAW_RATE)

    # -----------------------------------------------------
    # PREVIOUS ACTION
    # -----------------------------------------------------
    prev_action = np.asarray(prev_action, dtype=np.float32)

    obs = np.concatenate([
        lidar_norm,
        np.array([
            cruising_speed_norm,
            steering_norm,
            longitudinal_velocity_norm,
            lateral_velocity_norm,
            linear_velocity_norm,
            yaw_rate_norm,
            car_heading_norm,
            road_heading_norm,
            velocity_heading_norm,
            heading_error_deg_norm,
            slip_angle_deg_norm,
            dist2centerline_point_norm,
        ], dtype=np.float32),
        prev_action,
        np.array([
            cruising_speed,
            longitudinal_velocity,
            lateral_velocity,
            linear_velocity,
            slip_angle_deg,
            heading_error_deg,
            dist2centerline_point,
            yaw_rate,
        ])
    ])

    return obs.astype(np.float32)


def apply_action(driver: Driver, steering: float, throttle: float, brake: float, speed_ms: float = 0.0) -> None:
    """Send a normalised action to the Webots Driver.

    All three inputs are normalised:
        steering: [-1, 1]  (left negative, right positive)
        throttle: [ 0, 1]  (direct torque input via setThrottle)
        brake:    [ 0, 1]

    Uses setThrottle() + manual gear management so the internal PID of
    setCruisingSpeed() does not fight against the oversteer needed for drift.
    Gear shifts are triggered by RPM thresholds read from the Driver.
    """
    steering = max(-1.0, min(1.0, steering))
    throttle = max(0.0, min(1.0, throttle))
    brake = max(0.0, min(1.0, brake))

    prev_steer = getattr(apply_action, "_prev_steer", 0.0)
    alpha = 0.1
    steering = prev_steer + alpha * (steering - prev_steer)
    apply_action._prev_steer = steering

    driver.setSteeringAngle(steering)

    if brake > 0.1:
        driver.setThrottle(0.0)
        driver.setBrakeIntensity(brake)
    else:
        driver.setBrakeIntensity(0.0)
        driver.setThrottle(throttle)

        current_gear = driver.getGear()

        gear_ratios = [0.0, 12.5, 8.0, 5.35, 4.3, 4.0]
        wheel_radius = 0.36
        if current_gear > 0 and current_gear < len(gear_ratios):
            rpm = (speed_ms / wheel_radius) * gear_ratios[current_gear] * 60.0 / 6.2832
        else:
            rpm = 1000.0

        # Cooldown entre mudanças: 0.5s mínimo entre transições para evitar hunting.
        import time as _t
        last_shift = getattr(apply_action, "_last_shift_t", 0.0)
        can_shift = (_t.time() - last_shift) > 0.5

        if current_gear == 0:
            driver.setGear(1)
            apply_action._last_shift_t = _t.time()
        elif can_shift:
            # Histerese: gap largo entre shift-up (5000) e shift-down (1500).
            if rpm > 5000 and current_gear < 5:
                driver.setGear(current_gear + 1)
                apply_action._last_shift_t = _t.time()
                prev = getattr(apply_action, "_prev_gear", -1)
                if current_gear + 1 != prev:
                    # print(f"[gear] UP to {current_gear + 1} (rpm={rpm:.0f}, speed_ms={speed_ms:.1f})", flush=True)
                    apply_action._prev_gear = current_gear + 1
            elif rpm < 1500 and current_gear > 1:
                driver.setGear(current_gear - 1)
                apply_action._last_shift_t = _t.time()
                prev = getattr(apply_action, "_prev_gear", -1)
                if current_gear - 1 != prev:
                    # print(f"[gear] DOWN to {current_gear - 1} (rpm={rpm:.0f}, speed_ms={speed_ms:.1f})", flush=True)
                    apply_action._prev_gear = current_gear - 1


# max_episode_steps tells Gymnasium to automatically truncate an episode after 10,000 calls to step().
# If the Webots timestep is self.timestep = 32 then 10000 × 0.032s = 320s.
# So the maximum episode length is about 5.3 minutes.
# When the limit is reached, truncated = True is returned automatically by Gymnasium's wrapper.
register(
    id="Vehicle-v0",
    entry_point="vehicle_env:VehicleEnv",
    max_episode_steps=5000
)

class VehicleEnv(gym.Env):
    def __init__(self, mode="train", other_logs_filepath=None):
        self.global_timesteps = 0
        self.mode = mode
        if self.mode == "train":
            self.other_logs_filepath = other_logs_filepath

        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())
        print(f"Timestep: {self.timestep}")

        self.lidar = self.driver.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        #self.lidar.enablePointCloud()
        self.gyro = self.driver.getDevice("gyro")
        self.gyro.enable(self.timestep)
        self.gps = self.driver.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.driver.getDevice("compass")
        self.compass.enable(self.timestep)
        self.touch_front = self.driver.getDevice("touch sensor front")
        self.touch_front.enable(self.timestep)
        self.touch_left = self.driver.getDevice("touch sensor left")
        self.touch_left.enable(self.timestep)
        self.touch_right = self.driver.getDevice("touch sensor right")
        self.touch_right.enable(self.timestep)

        self.vehicle_node = self.driver.getFromDef("VEHICLE")

        # TRACK
        self.world_name = os.path.basename(self.driver.getWorldPath())
        self.centerline = get_centerline(self.world_name)
        self.comulative_progress = compute_centerline_s(self.centerline)

        if "arena" in self.world_name:
            self._n_checkpoints = 6
            self._arena_cx = 150
            self._arena_cy = 0
            self.next_checkpoint = 0
            self.prev_sector = None

        self.episode_progress = 0

        # steering, throttle
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 12 for the lidar bins + 12 for the other sensors + 2 for the previous action
        self.obs_size = 12 + 12 + 2
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0]*12 + [0.0,-1.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0,-1.0,0.0],
                         dtype=np.float32),
            high=np.array([1.0]*self.obs_size, dtype=np.float32),
            dtype=np.float32
        )

        # Inital state of car to warp and reset
        self.initial_vehicle_position = self.vehicle_node.getField("translation").getSFVec3f()
        self.initial_vehicle_orientation = self.vehicle_node.getField("rotation").getSFRotation()
        self.inital_steering = 0.0
        self.inital_speed = self.driver.getCurrentSpeed()
        self.inital_throttle = self.driver.getThrottle()
        self.initial_brake_intensity = self.driver.getBrakeIntensity()
        self.initial_gear = self.driver.getGear()

        # Some variables
        self.previous_easier_control = False
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

        # Episode variables
        self.episode_num_timesteps = 0
        self.reward_episode = 0
        self.num_epochs = 0
        self.resetting = False

        self.episode_penalty_reward = 0
        self.episode_going_straight_reward = 0
        self.episode_slip_angle_reward = 0
        self.episode_heading_error_angle_reward = 0

        # Other logs
        self.update_logs_freq = 75000
        self.logs = {
            "episode_penalty_reward": [],
            "episode_going_straight_reward": [],
            "episode_slip_angle_reward": [],
            "episode_heading_error_angle_reward": []
        }
        self._logs = {
            "episode_penalty_reward": [],
            "episode_going_straight_reward": [],
            "episode_slip_angle_reward": [],
            "episode_heading_error_angle_reward": []
        }

        self.next_logs_update_step = self.update_logs_freq

    def full_warp_vehicle(self, new_position, new_orientation) -> None:
        trans_field = self.vehicle_node.getField("translation")
        trans_field.setSFVec3f(new_position)
        rot_field = self.vehicle_node.getField("rotation")
        rot_field.setSFRotation(new_orientation)
        self.vehicle_node.resetPhysics()
        time.sleep(0.2)

    def _get_arena_sector(self, x: float, y: float) -> int:
        """Maps (x, y) to one of N evenly-spaced sectors around the arena centre."""
        angle = math.atan2(y - self._arena_cy, x - self._arena_cx)
        angle = (angle + 2 * math.pi) % (2 * math.pi)  # normalise to [0, 2π]
        return int(angle / (2 * math.pi / self._n_checkpoints)) % self._n_checkpoints

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.driver.setSteeringAngle(self.inital_steering)
        self.driver.setCruisingSpeed(self.inital_speed)
        self.driver.setThrottle(self.inital_throttle)
        self.driver.setBrakeIntensity(self.initial_brake_intensity)
        self.driver.setGear(self.initial_gear)

        # Warp robot to initial position
        self.driver.step()
        self.full_warp_vehicle(self.initial_vehicle_position, self.initial_vehicle_orientation)
        obs = build_observation(driver=self.driver, centerline=self.centerline, lidar=self.lidar, gyro=self.gyro,
                                 gps=self.gps, compass=self.compass,prev_action=self.prev_action, type=self.type)

        print(f"Reward: {self.reward_episode} | Time: {round(self.episode_num_timesteps*(self.timestep/1000), 2)}s")
        self.episode_num_timesteps = 0
        self.reward_episode = 0
        self.num_epochs += 1
        self.resetting = True

        if "arena" in self.world_name:
            x0, y0 = self.vehicle_node.getField("translation").getSFVec3f()[:2]
            start_sector = self._get_arena_sector(x0, y0)
            self.prev_sector = start_sector
            self.next_checkpoint = (start_sector + 1) % self._n_checkpoints

        print(
            f"Episode Progress: {self.episode_progress}\n"
            f"Close to wall Penalty: {self.episode_penalty_reward}\n"
            f"Going straight reward: {self.episode_going_straight_reward}\n"
            f"Slip angle reward: {self.episode_slip_angle_reward}\n"
            f"Heading error angle reward: {self.episode_heading_error_angle_reward}\n"
        )

        if self.mode == "train":
            self._logs["episode_penalty_reward"].append(self.episode_penalty_reward)
            self._logs["episode_going_straight_reward"].append(self.episode_going_straight_reward)
            self._logs["episode_slip_angle_reward"].append(self.episode_slip_angle_reward)
            self._logs["episode_heading_error_angle_reward"].append(self.episode_heading_error_angle_reward)

        self.episode_progress = 0
        self.episode_penalty_reward = 0
        self.episode_going_straight_reward = 0
        self.episode_slip_angle_reward = 0
        self.episode_heading_error_angle_reward = 0

        if self.mode == "train" and self.global_timesteps > self.next_logs_update_step:
            self.next_logs_update_step += self.update_logs_freq
            self.logs["episode_penalty_reward"].append((float(round(np.mean(self._logs[
            "episode_penalty_reward"]), 3)),self.global_timesteps))
            self.logs["episode_going_straight_reward"].append((float(round(np.mean(self._logs[
            "episode_going_straight_reward"]),3)), self.global_timesteps))
            self.logs["episode_slip_angle_reward"].append((float(round(np.mean(self._logs[
            "episode_slip_angle_reward"]),3)), self.global_timesteps))
            self.logs["episode_heading_error_angle_reward"].append((float(round(np.mean(self._logs[
            "episode_heading_error_angle_reward"]),3)), self.global_timesteps))
            self._logs["episode_penalty_reward"] = []
            self._logs["episode_going_straight_reward"] = []
            self._logs["episode_slip_angle_reward"] = []
            self._logs["episode_heading_error_angle_reward"] = []
            with open(self.other_logs_filepath, "w") as f:
                json.dump(self.logs, f, indent=4)

        return obs[:self.obs_size], {}


    def compute_reward(self,obs = None, steering=0.0, throttle=0.0):
        terminated = False
        truncated = False
        reward = 0
        cruising_speed = obs[-8]
        slip_angle_deg = obs[-4]
        heading_error_deg = obs[-3]
        dist2centerline_point = obs[-2]

        x, y = self.vehicle_node.getField("translation").getSFVec3f()[:2]

        v_front = self.touch_front.getValue()
        v_left = self.touch_left.getValue()
        v_right = self.touch_right.getValue()
        touch = float(max([v_front, v_left, v_right]))

        if "arena" in self.world_name:
            if touch > 0:
                if self.resetting:
                    self.resetting = False
                else:
                    reward -= 100
                    terminated = True
            elif dist2centerline_point >= 5:
                reward -= 7.5
                print("Penalty")
            else:
                reward += 0.2
                current_sector = self._get_arena_sector(x, y)
                if current_sector != self.prev_sector:
                    prev_expected = (self.next_checkpoint - 1) % self._n_checkpoints
                    if (current_sector == self.next_checkpoint and
                            self.prev_sector == prev_expected):
                        reward += 30
                        self.next_checkpoint = (self.next_checkpoint + 1) % self._n_checkpoints
                        angle_deg = round(math.degrees(math.atan2(y - self._arena_cy, x - self._arena_cx)) % 360)
                        print(f"[checkpoint {current_sector}/{self._n_checkpoints}] next={self.next_checkpoint} angle={angle_deg}°")
                    self.prev_sector = current_sector

                if cruising_speed > 25:
                    if 30 < heading_error_deg < 60:
                        print("Drift")
                        reward += 50
                    elif heading_error_deg < 30 and heading_error_deg > -90 and steering < 0:
                        print("Steering left to drift")
                        reward += 10
                    elif heading_error_deg > 60 and heading_error_deg < 90 and steering > 0:
                        print("Steering right to maintain drift")
                        reward += 10
        elif "track" in self.world_name:
            car_pos = self.driver.getFromDef("VEHICLE").getField("translation").getSFVec3f()
            idx, _, _ = closest_centerline_point(car_pos[:2], self.centerline)
            progress = self.comulative_progress[idx]

            if touch > 0:
                if self.resetting:
                    self.resetting = False
                else:
                    reward -= 100
                    terminated = True
            elif dist2centerline_point >= 5:
                reward += -2
                self.episode_penalty_reward += -2
            else:
                if x < 130 or x > 170:
                    if slip_angle_deg > 30 and slip_angle_deg < 60:
                        reward += 10
                        self.episode_slip_angle_reward += 50
                    if heading_error_deg > 30 and heading_error_deg < 60:
                        reward += 5
                        self.episode_heading_error_angle_reward += 50
                if x > 130 and x < 170:
                    if abs(slip_angle_deg) < 10 and abs(heading_error_deg) < 10 and progress > self.episode_progress:
                        reward += 2
                        self.episode_going_straight_reward += 1

            if progress > self.episode_progress:
                self.episode_progress = progress

        return reward, terminated, truncated


    def step(self, action):
        steering = float(action[0])
        throttle = float(action[1])

        apply_action(self.driver, steering, throttle, 0.0, speed_ms=abs(self.driver.getCurrentSpeed()))
        self.previous_easier_control = facilitate_driving(self.driver, self.previous_easier_control, self.world_name)
        self.driver.step()

        obs = build_observation(driver=self.driver, centerline=self.centerline, lidar=self.lidar, gyro=self.gyro,
                                gps=self.gps, compass=self.compass, prev_action=self.prev_action, type=self.type)

        self.prev_action = np.array([steering, throttle], dtype=np.float32)

        #reward, terminated, truncated = self.compute_reward(type="complex", obs=obs, steering=steering, throttle=throttle)
        reward, terminated, truncated = self.compute_reward(obs=obs, steering=steering, throttle=throttle)

        self.reward_episode += reward
        self.episode_num_timesteps += 1

        self.global_timesteps += 1

        return (obs[:self.obs_size], reward, terminated, truncated, {})