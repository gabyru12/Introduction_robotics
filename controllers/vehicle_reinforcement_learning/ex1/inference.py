import os
import json

import gymnasium as gym
from stable_baselines3 import PPO

import controllers.vehicle_reinforcement_learning.ex1.vehicle_env

MODE = "inference"

def main():
    base_dir = os.path.join(os.getcwd(), "logs")
    metrics_save_path = os.path.join(base_dir, "eval_metrics", f"inference_metrics.json")

    model = PPO.load(base_dir +
                     "/drift_12h_norm_rewards_timestep_32_facilitate_drifting_130_170_progress_reward_01_checkpoints"
                     "/ppo_vehicle_1200000_steps.zip")
    env = gym.make("Vehicle-v0", mode=MODE)

    metrics = []
    episode = 1
    step = 0

    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        row = {
            "evalback": 12,
            "episode": episode,
            "step": step,
        }
        row.update(info)
        metrics.append(row)

        if terminated or truncated:
            with open(metrics_save_path, "w") as f:
                json.dump(metrics, f, indent=2)
            episode += 1
            step = 0
            obs, info = env.reset()

if __name__ == '__main__':
    main()