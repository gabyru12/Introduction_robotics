import time

import gymnasium as gym
import numpy as np
import os
import sys

from vehicle import Driver
from .sensors import VehicleSensors
from .my_controller import apply_action

from gymnasium.envs.registration import register

#_conda_lib_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
#if os.path.isdir(_conda_lib_bin):
#    if hasattr(os, "add_dll_directory"):   # Python 3.8+ Windows only
#        os.add_dll_directory(_conda_lib_bin)
#    # Also prepend to PATH as a belt-and-braces fallback.
#    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")

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
        root = self.driver.getRoot()
        children = root.getField("children")
        self.vehicle_node = None

        for i in range(children.getCount()):
            node = children.getMFNode(i)
            if node.getDef() == "VEHICLE":
                self.vehicle_node = node
                break

        self.timestep = int(self.driver.getBasicTimeStep())
        print(f"Timestep: {self.timestep}")
        self.sensors = VehicleSensors(self.driver, self.timestep)

        # TODO: Set up better action_space
        # steering, throttle and break
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 32 for the lidar bins + 5 for the other sensors
        obs_size = 32 + 5

        # TODO: set a low and high value for each of the lidar measurements and the other 5 sensors
        # TODO: find better normalised values for each metric and find better ways of processing metric information
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.initial_vehicle_position = self.vehicle_node.getField("translation").getSFVec3f()
        self.initial_vehicle_orientation = self.vehicle_node.getField("rotation").getSFRotation()
        self.inital_steering = 0
        self.inital_speed = self.driver.getCurrentSpeed()
        self.inital_throttle = self.driver.getThrottle()
        self.initial_brake_intensity = self.driver.getBrakeIntensity()
        self.initial_gear = self.driver.getGear()

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
        time.sleep(1)


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
        obs = self.sensors.read(self.driver.getTime())

        print(f"Reward: {self.reward_episode} | Timesteps: {round(self.num_timesteps*(self.timestep/1000), 2)}s")
        self.num_timesteps = 0
        self.reward_episode = 0
        self.num_epochs += 1
        self.resetting = True

        return self.sensors.to_rl_vector(obs), {}

    def step(self, action):
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])

        obs = self.sensors.read(self.driver.getTime())
        apply_action(self.driver, steer, throttle, brake, speed_ms=abs(obs.forward_speed))
        self.driver.step()

        observation = self.sensors.to_rl_vector(obs)

        terminated = False
        truncated = False

        reward = 0

        # TODO: Compute better reward
        if obs.touch > 0:
            if self.resetting:
                self.resetting = False
            else:
                reward -= 1000000
                terminated = True
        elif obs.forward_speed < 0:
            reward -= 5
        elif obs.forward_speed > 0:
            reward = obs.forward_speed

        if self.num_timesteps % 100 == 0:
            print(obs.lidar_ranges, obs.lidar_max_range)
        self.num_timesteps += 1
        self.reward_episode += reward

        return (observation, reward, terminated, truncated, {})