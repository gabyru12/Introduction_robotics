import os
import time

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

import controllers.vehicle_reinforcement_learning.ex1.vehicle_env


# ---------------------------------------------------------------------
# DQN só aceita action spaces discretos. O env Vehicle-v0 usa um Box
# contínuo (steering, throttle), por isso convertemos cada inteiro
# discreto num par (steering, throttle) com este wrapper.
# O env original fica intacto -> o training_ppo.py (PPO) continua a funcionar.
# ---------------------------------------------------------------------

# steering em [-1, 1] (negativo = esquerda, positivo = direita)
# throttle em [0, 1]
STEERING_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
THROTTLE_VALUES = [0.0, 0.33, 0.66, 1.0]

# Grelha completa: 5 steering x 4 throttle = 20 ações discretas
DISCRETE_ACTIONS = [
    (s, t)
    for s in STEERING_VALUES
    for t in THROTTLE_VALUES
]


class DiscretizeAction(gym.ActionWrapper):
    """Expõe Discrete(N) ao agente e traduz para o Box(steering, throttle)."""

    def __init__(self, env, actions=DISCRETE_ACTIONS):
        super().__init__(env)
        self._actions = [np.array(a, dtype=np.float32) for a in actions]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, act):
        return self._actions[int(act)]


def main() -> None:
    base_dir: str = os.path.join(os.getcwd(), "logs")

    env = gym.make("Vehicle-v0")
    env = DiscretizeAction(env)

    model = DQN(
        "MlpPolicy", env, verbose=1,
        exploration_fraction=0.4,
        exploration_final_eps=0.1,
        tensorboard_log=base_dir + "/tensorboard_logs",
    )

    time_str: str = time.strftime("%Y%m%d-%H%M%S")
    eval_callback = EvalCallback(
        env,
        n_eval_episodes=1,
        eval_freq=40000,
        best_model_save_path=base_dir + "/best_model_dqn",
        log_path=base_dir + "/eval_logs_results_dqn",
    )

    model.learn(
        total_timesteps=3_300_000,
        log_interval=10,
        tb_log_name="dqn_" + time_str,
        callback=eval_callback,
    )


if __name__ == "__main__":
    main()
