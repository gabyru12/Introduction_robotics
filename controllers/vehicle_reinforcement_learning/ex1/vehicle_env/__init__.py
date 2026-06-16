import time

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import math

from vehicle import Driver
from .my_controller import apply_action


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


def get_centerline():
    # =====================================================
    # Segment 1 (bottom straight)
    # =====================================================
    seg1 = line_points(
        135, -25,
        180, -25,
        n=50
    )

    # =====================================================
    # Segment 2 (right semicircle)
    # center = (30, 0)
    # radius = 25
    # from bottom to top
    # =====================================================
    seg2 = arc_points(
        cx=180,
        cy=0,
        radius=25,
        start_angle=-np.pi/2,
        end_angle=np.pi/2,
        n=100
    )

    # =====================================================
    # Segment 3 (top straight)
    # =====================================================
    seg3 = line_points(
        180, 25,
        120, 25,
        n=100
    )

    # =====================================================
    # Segment 4 (left semicircle)
    # center = (-30, 0)
    # radius = 25
    # from top to bottom
    # =====================================================
    seg4 = arc_points(
        cx=120,
        cy=0,
        radius=25,
        start_angle=np.pi/2,
        end_angle=3*np.pi/2,
        n=100
    )

    # =====================================================
    # Segment 5 (bottom straight)
    # =====================================================
    seg5 = line_points(
        120, -25,
        135, -25,
        n=50
    )

    # =====================================================
    # Complete centerline
    # =====================================================
    centerline = np.vstack([
        seg1[:-1],
        seg2[:-1],
        seg3[:-1],
        seg4[:-1],
        seg5
    ])

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


def facilitate_driving(driver, previous_easier_control):
    car_node = driver.getFromDef("VEHICLE")
    translation_field = car_node.getField("translation")
    x = translation_field.getSFVec3f()[0]
    just_turned_easier_control_on = False
    just_turned_easier_control_off = False
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
    MAX_CRUISING_SPEED=250.0,
    MAX_LINEAR_VELOCITY=30.0,
    MAX_LATERAL_VELOCITY=30.0,
    MAX_YAW_RATE=6.0,
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
    lidar_norm = np.clip(lidar_bins / lidar.getMaxRange(),0.0,1.0)

    # -----------------------------------------------------
    # VEHICLE STATE (NECESSARY)
    # -----------------------------------------------------
    cruising_speed = driver.getCurrentSpeed()
    cruising_speed_norm = normalize_unsigned(abs(cruising_speed), MAX_CRUISING_SPEED)
    steering_angle = driver.getSteeringAngle()
    steering_norm = normalize_signed(steering_angle, MAX_STEER_ANGLE)

    # -----------------------------------------------------
    # GPS
    # -----------------------------------------------------
    gps_velocity_vector = np.asarray(gps.getSpeedVector(), dtype=np.float32)

    # -----------------------------------------------------
    # LINEAR VELOCITY & LATERAL VELOCITY
    # -----------------------------------------------------
    north = compass.getValues()
    heading = math.atan2(-float(north[0]), float(north[1]))
    heading_unit = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=np.float32)
    right_unit = np.array([-math.sin(heading), math.cos(heading), 0.0], dtype=np.float32)
    linear_velocity = float(np.dot(gps_velocity_vector, heading_unit))
    linear_velocity_norm = normalize_unsigned(linear_velocity, MAX_LINEAR_VELOCITY)
    lateral_velocity = float(np.dot(gps_velocity_vector, right_unit))
    lateral_velocity_norm = normalize_unsigned(lateral_velocity, MAX_LATERAL_VELOCITY)

    # -----------------------------------------------------
    # DISTANCE TO CENTERLINE & DRIFT ANGLE
    # -----------------------------------------------------
    car_pos = driver.getFromDef("VEHICLE").getField("translation").getSFVec3f()
    idx, centerline_point, dist2_centerline_point = closest_centerline_point(car_pos[:2], centerline)
    road_heading = centerline[(idx + 1) % len(centerline)] - centerline[idx]
    road_heading = math.atan2(road_heading[1], road_heading[0])
    road_heading_unit = np.array([round(math.cos(road_heading),3), round(math.sin(road_heading),3), 0.0], dtype=np.float32)

    north = compass.getValues()
    car_heading = -math.atan2(-float(north[0]), float(north[1]))
    car_heading_unit = np.array([round(math.cos(car_heading),3), round(math.sin(car_heading),3), 0.0], dtype=np.float32)

    drift_angle = int(_wrap_angle(car_heading - road_heading) * 180 / math.pi)
    dist2centerline_point = dist2_centerline_point
    car_heading = int(car_heading * 180 / math.pi)
    road_heading = int(road_heading * 180 / math.pi)
    car_heading_norm = normalize_signed(car_heading, MAX_CAR_HEADING_ANGLE)
    road_heading_norm = normalize_signed(road_heading, MAX_ROAD_HEADING_ANGLE)
    drift_angle_norm = normalize_signed(drift_angle, MAX_DRIFT_ANGLE)
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
            linear_velocity_norm,
            lateral_velocity_norm,
            yaw_rate_norm,
            car_heading_norm,
            road_heading_norm,
            drift_angle_norm,
            dist2centerline_point_norm,
        ], dtype=np.float32),
        prev_action,
        np.array([
            cruising_speed_norm,
            linear_velocity,
            lateral_velocity,
            drift_angle,
            dist2centerline_point,
            yaw_rate_norm,
        ])
    ])

    return obs.astype(np.float32)

# max_episode_steps tells Gymnasium to automatically truncate an episode after 10,000 calls to step().
# If the Webots timestep is self.timestep = 32 then 10000 × 0.032s = 320s.
# So the maximum episode length is about 5.3 minutes.
# When the limit is reached, truncated = True is returned automatically by Gymnasium's wrapper.
register(
    id="Vehicle-v0",
    entry_point="vehicle_env:VehicleEnv",
    max_episode_steps=10000
)

class VehicleEnv(gym.Env):
    def __init__(self):
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
        # root = self.driver.getRoot()
        # children = root.getField("children")
        # self.vehicle_node = None
        #
        # for i in range(children.getCount()):
        #     node = children.getMFNode(i)
        #     if node.getDef() == "VEHICLE":
        #         self.vehicle_node = node
        #         break

        # TRACK
        self.centerline = get_centerline()
        self.comulative_progress = compute_centerline_s(self.centerline)

        # steering, throttle
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 12 for the lidar bins + 11 for the other sensors + 2 for the previous action
        obs_size = 12 + 9 + 2

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0]*12 + [0.0,-1.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,0.0,-1.0,0.0], dtype=np.float32),
            high=np.array([1.0]*obs_size, dtype=np.float32),
            dtype=np.float32
        )

        self.initial_vehicle_position = self.vehicle_node.getField("translation").getSFVec3f()
        self.initial_vehicle_orientation = self.vehicle_node.getField("rotation").getSFRotation()
        self.inital_steering = 0.0
        self.inital_speed = self.driver.getCurrentSpeed()
        self.inital_throttle = self.driver.getThrottle()
        self.initial_brake_intensity = self.driver.getBrakeIntensity()
        self.initial_gear = self.driver.getGear()

        self.previous_easier_control = False
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

        self.num_timesteps = 0
        self.reward_episode = 0
        self.num_epochs = 0
        self.resetting = False

    def full_warp_vehicle(self, new_position, new_orientation) -> None:
        trans_field = self.vehicle_node.getField("translation")
        trans_field.setSFVec3f(new_position)
        rot_field = self.vehicle_node.getField("rotation")
        rot_field.setSFRotation(new_orientation)
        self.vehicle_node.resetPhysics()
        time.sleep(0.2)


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
        obs = build_observation(self.driver, self.centerline, self.lidar, self.gyro, self.gps, self.compass, self.prev_action)

        print(f"Reward: {self.reward_episode} | Time: {round(self.num_timesteps*(self.timestep/1000), 2)}s")
        self.num_timesteps = 0
        self.reward_episode = 0
        self.num_epochs += 1
        self.resetting = True

        return obs[:23], {}


    def step(self, action):
        steering = float(action[0])
        throttle = float(action[1])

        obs = build_observation(self.driver, self.centerline, self.lidar, self.gyro, self.gps, self.compass, self.prev_action)
        apply_action(self.driver, steering, throttle, 0.0, speed_ms=abs(self.driver.getCurrentSpeed()))
        self.prev_action = np.array([steering, throttle], dtype=np.float32)
        self.previous_easier_control = facilitate_driving(self.driver, self.previous_easier_control)
        self.driver.step()

        terminated = False
        truncated = False

        reward = 0
        cruising_speed = obs[-6]
        linear_velocity = obs[-5]
        drift_angle = obs[-3]
        dist2centerline_point = obs[-2]
        x, y = self.vehicle_node.getField("translation").getSFVec3f()[:2]


        # TODO: Compute better reward
        v_front = self.touch_front.getValue()
        v_left = self.touch_left.getValue()
        v_right = self.touch_right.getValue()
        touch = float(max([v_front, v_left, v_right]))
        if touch > 0:
            if self.resetting:
                self.resetting = False
            else:
                reward -= 100
                terminated = True
        elif dist2centerline_point >= 5:
            reward -= 10
            print("Penality")
        else:
            if x < 130 or x > 170:
                if drift_angle > 30 and drift_angle < 60 and cruising_speed > 100:
                    print("Drift")
                    reward += 50
                if drift_angle < 30 and drift_angle > -90 and steering < 0:
                    print("Steering left to drift")
                    reward += 10
                elif drift_angle > 60 and drift_angle < 90 and steering > 0:
                    print("Steering right to maintain drift")
                    reward += 10
            if x > 130 and x < 170:
                if abs(drift_angle) < 30 and cruising_speed > 30:
                    print("Going straight")
                    reward += 1

        self.reward_episode += reward
        self.num_timesteps += 1

        return (obs[:23], reward, terminated, truncated, {})