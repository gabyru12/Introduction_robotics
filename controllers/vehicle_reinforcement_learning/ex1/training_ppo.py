import os
import time
import json

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import controllers.vehicle_reinforcement_learning.ex1.vehicle_env

class TrackingEvalCallback(EvalCallback):
    """
    EvalCallback normal, mas que durante cada avaliação (evalback) guarda,
    a cada step de cada um dos n_eval_episodes, todas as métricas devolvidas
    no info dict do env (drift_angle, slip_angle, yaw_rate, coordenadas, etc)
    num ficheiro JSON.

    O SB3 só chama `_log_success_callback` durante a evaluate_policy() corrida
    pelo EvalCallback, ou seja, durante o treino normal este método nunca é
    invocado -> serve exatamente de "flag" para sabermos que estamos num evalback.
    """
    def __init__(self, *args, metrics_save_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_save_path = metrics_save_path
        os.makedirs(os.path.dirname(self.metrics_save_path), exist_ok=True)
        self.all_metrics: list[dict] = []
        self._eval_count = 0
        self._episode_in_eval = 0
        self._step_in_episode = 0
        self._last_done = True

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        info = locals_["info"]
        done = locals_["done"]
        if "terminal_observation" in info:
            del info["terminal_observation"]
        if self._last_done:
            self._episode_in_eval += 1
            self._step_in_episode = 0
        self._last_done = done
        self._step_in_episode += 1
        row = {
            "evalback": self._eval_count,
            "episode": self._episode_in_eval,
            "step": self._step_in_episode,
        }
        row.update(info)
        self.all_metrics.append(row)

    def _on_step(self):
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if is_eval_step:
            self._eval_count += 1
            self._episode_in_eval = 0
            self._last_done = True
        continue_training = super()._on_step()
        if is_eval_step:
            with open(self.metrics_save_path, "w") as f:
                json.dump(self.all_metrics, f, indent=2)
        return continue_training


time_str: str = time.strftime("%Y%m%d-%H%M%S")
learn_new = True

def main() -> None:
    base_dir: str = os.path.join(os.getcwd(), "logs")

    env = gym.make("Vehicle-v0", mode="train")
    # In the tensorboard_logs directory, run
    # python -m tensorboard.main --logdir "./"

    if learn_new:
        model = PPO(
            "MlpPolicy", env , verbose=1,
            # learning_rate=3e-3,
            tensorboard_log=base_dir + '/tensorboard_logs',
        )
        eval_callback = TrackingEvalCallback(
            env,
            n_eval_episodes=5,
            eval_freq=100000,
            best_model_save_path=base_dir + "/best_model",
            log_path=base_dir + "/eval_logs_results",
            metrics_save_path=os.path.join(base_dir, "eval_metrics", f"{time_str}.json"),
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,  # 13min
            save_path="logs/checkpoints/",
            name_prefix="ppo_vehicle"
        )
        model.learn(
            total_timesteps=7_650_000,  # 17h
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
            eval_freq=100000,
            best_model_save_path=base_dir + "/best_model",
            log_path=base_dir + "/eval_logs_results"
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path="new_logs/checkpoints/drift_11h_arena_norm_reward_8192_n_steps_checkpoints/",
            name_prefix="ppo_vehicle"
        )
        model.learn(
            total_timesteps=9_000_000,  # additional timesteps
            log_interval=10,
            tb_log_name="time_str",
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False,
        )

if __name__ == '__main__':
    main()
