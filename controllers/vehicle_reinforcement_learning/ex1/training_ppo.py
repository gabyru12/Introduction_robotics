import os
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

import controllers.vehicle_reinforcement_learning.ex1.vehicle_env

MODE = "train"
time_str: str = time.strftime("%Y%m%d-%H%M%S")
OTHER_LOGS_FILENAME = f"other_logs_{time_str}.json"
learn_new = True

def main() -> None:
    base_dir: str = os.path.join(os.getcwd(), "logs")
    other_logs_dirpath = os.path.join(base_dir, "other_logs")
    os.makedirs(other_logs_dirpath, exist_ok=True)
    other_logs_filepath = os.path.join(other_logs_dirpath, OTHER_LOGS_FILENAME)

    env = gym.make("Vehicle-v0", mode=MODE, other_logs_filepath=other_logs_filepath)
    # In the tensorboard_logs directory, run
    # python -m tensorboard.main --logdir "./"
    if learn_new:
        model = PPO(
            "MlpPolicy", env, verbose=1,
            #learning_rate=3e-3,
            tensorboard_log=base_dir + '/tensorboard_logs',
        )
        eval_callback = EvalCallback(
            env,
            n_eval_episodes=5,
            eval_freq=400000,
            best_model_save_path=base_dir + "/best_model",
            log_path=base_dir + "/eval_logs_results"
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path="logs/drift_open_space_reward_continuo_checkpoints/",
            name_prefix="ppo_vehicle"
        )
        model.learn(
            total_timesteps=10000000 * 50000,
            log_interval=10,
            tb_log_name=time_str,
            callback=[eval_callback, checkpoint_callback]
        )
    else:
        checkpoint_path = "logs/drift_open_space_reward_continuo_checkpoints/ppo_vehicle_300000_steps.zip"
        model = PPO.load(
            checkpoint_path,
            env=env,
            tensorboard_log=base_dir + '/tensorboard_logs',
        )
        eval_callback = EvalCallback(
            env,
            n_eval_episodes=5,
            eval_freq=400000,
            best_model_save_path=base_dir + "/best_model",
            log_path=base_dir + "/eval_logs_results"
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path="logs/drift_open_space_reward_continuo_checkpoints/",
            name_prefix="ppo_vehicle"
        )
        model.learn(
            total_timesteps=10000000 * 50000,  # additional timesteps
            log_interval=10,
            tb_log_name="drift_open_space_reward_continuo_not_in_scientific_paper",
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False,
        )

if __name__ == '__main__':
    main()
