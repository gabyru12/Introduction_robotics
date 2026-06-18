import os

import gymnasium as gym
from stable_baselines3 import PPO

def main() -> None:
    base_dir: str = os.path.join(os.getcwd(), "logs")
    # Change to the .zip file of your own model!
    model = PPO.load(base_dir + "/best_model/best_model.zip")
    env = gym.make("Vehicle-v0")

    obs, _info = env.reset() # reset the environment
    while True:
        action, _states = model.predict(obs) # predict the next action
        obs, reward, terminated, truncated, _info = env.step(action[0]) # step the environment
        if truncated:
            obs, _info = env.reset() # reset the environment

if __name__ == '__main__':
    main()