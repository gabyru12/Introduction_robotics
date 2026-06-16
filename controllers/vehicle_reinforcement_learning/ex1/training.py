import os
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import controllers.vehicle_reinforcement_learning.ex1.vehicle_env

def main() -> None:
    base_dir: str = os.path.join(os.getcwd(), "logs")
    env = gym.make("Vehicle-v0")
    # In the tensorboard_logs directory, run
    # python -m tensorboard.main --logdir "./"
    model = PPO(
        "MlpPolicy", env, verbose=1,
        #learning_rate=3e-3,
        tensorboard_log=base_dir + '/tensorboard_logs',
    )
    time_str: str = time.strftime("%Y%m%d-%H%M%S")
    eval_callback = EvalCallback(
        env,
        n_eval_episodes=5,
        eval_freq=40000,
        best_model_save_path=base_dir + "/best_model",
        log_path=base_dir + "/eval_logs_results"
    )
    model.learn(
        total_timesteps=10000000 * 50000,
        log_interval=10,
        tb_log_name=time_str,
        callback=eval_callback
    )

if __name__ == '__main__':
    main()
