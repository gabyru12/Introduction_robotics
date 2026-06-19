import os

import gymnasium as gym
from stable_baselines3 import PPO

import controllers.vehicle_reinforcement_learning.ex1.vehicle_env

MODE = "inference"
TYPE = "simple"

def main() -> None:
    base_dir: str = os.path.join(os.getcwd(), "logs")
    # Change to the .zip file of your own model!
    model = PPO.load(base_dir + "/drift_open_space_6h_30_60_degrees_checkpoints/ppo_vehicle_2500000_steps.zip")
    env = gym.make("Vehicle-v0", mode=MODE, type=TYPE)

    obs, _info = env.reset() # reset the environment
    while True:
        action, _states = model.predict(obs) # predict the next action
        obs, reward, terminated, truncated, _info = env.step(action) # step the environment
        if truncated:
            obs, _info = env.reset() # reset the environment

if __name__ == '__main__':
    main()