# Autonomous Drifting Using Reinforcement Learning in Webots

Project developed for **Introduction to Intelligent Robotics (CC3046)**.

**Authors**
- André Amaral
- Gabriel Oliveira
- José Sousa

---

# Overview

This project investigates the use of **Deep Reinforcement Learning (DRL)** to train an autonomous vehicle capable of performing sustained drifting manoeuvres inside the **Webots** simulator.

Two reinforcement learning algorithms were evaluated:

- **Proximal Policy Optimization (PPO)**
- **Deep Q-Network (DQN)**

The final solution uses PPO to learn controlled drifting behaviour using LiDAR and vehicle-state information.

---

# Tested Environment

The project was developed and tested using:

- Windows 11
- Webots R2025a
- Python 3.14
- PyCharm Professional / Community Edition

Compatibility with Linux and macOS has not been verified.

---

# Required Software

## 1. Webots

Install:

**Webots R2025a**

Recommended installation path:

```text
C:\Program Files\Webots\
```

---

## 2. Python

Install:

```text
Python 3.14
```

---

## 3. PyCharm

Install:

```text
JetBrains PyCharm
```

Community or Professional Edition may be used.

---

# Python Dependencies

Install the required packages:

```bash
pip install gymnasium stable-baselines3 tensorboard
```

---

# Project Setup

## Open the Project

Extract the repository and open it in PyCharm.

Example:

```text
C:\autonomous_drifting\
```

---

## Configure Python Interpreter

1. Open PyCharm
2. Select:

```text
Add New Interpreter
```

3. Choose:

```text
Local Interpreter
```

4. Select Python 3.14

---

## Configure Webots Python API

Add the following folder as a project content root:

```text
C:\Program Files\Webots\lib\controller\python
```

This allows PyCharm to correctly resolve:

```python
from controller import Robot
from vehicle import Driver
```

and other Webots APIs.

---

# Running the Simulation

## Open Webots

Launch Webots and open one of the available worlds:

```text
worlds/arena.wbt
```

or

```text
worlds/track.wbt
```

---

# PPO Training

Open:

```text
controllers/
└── vehicle_reinforcement_learning/
    └── ex1/
        └── vehicle_env/
            └── training_ppo.py
```

To start training from scratch:

```python
learn_new = True
```

Run:

```text
training_ppo.py
```

The script will:

- Create the Gymnasium environment
- Train a PPO agent
- Save checkpoints
- Log TensorBoard statistics
- Run evaluation callbacks

---

# Continue PPO Training

To continue training from a checkpoint:

```python
learn_new = False
```

and specify:

```python
checkpoint_path = "logs/saved_checkpoints/ppo_vehicle_xxxxx_steps.zip"
```

Run:

```text
training_ppo.py
```

again.

---

# DQN Training

Open:

```text
training_dqn.py
```

and run it.

The DQN implementation discretizes the continuous control space into steering/throttle combinations before training.

---

# Running a Trained Agent

Open:

```text
inference.py
```

Load the desired checkpoint:

```python
model = PPO.load(
    "logs/saved_checkpoints/<checkpoint_folder>/ppo_vehicle_xxxxx_steps.zip"
)
```

Run:

```text
inference.py
```

The vehicle will execute the learned policy inside Webots.

---

# TensorBoard

Training statistics are stored in:

```text
logs/tensorboard_logs/
```

Launch TensorBoard:

```bash
python -m tensorboard.main --logdir "./logs/tensorboard_logs"
```

Open:

```text
http://localhost:6006
```

in a browser.

---

# Project Structure

```text
Introduction_robotics/
│
├── controllers/
│   │
│   ├── test_folder_keyboard_control/
│   │   ├── main.py
│   │   └── rule_based.py
│   │
│   └── vehicle_reinforcement_learning/
│       └── ex1/
│           │
│           ├── logs/
│           │   ├── eval_logs_results/
│           │   ├── eval_metrics/
│           │   ├── saved_checkpoints/
│           │   └── tensorboard_logs/
│           │
│           └── vehicle_env/
│               ├── __init__.py
│               ├── training_ppo.py
│               ├── training_dqn.py
│               ├── inference.py
│               ├── plot_feature_all_laps.py
│               ├── plot_feature_and_lap.py
│               └── plot_track_drift.py
│
├── plugins/
│
├── protos/
│   └── BmwX5.proto
│   └── BmwX5Wheel.proto
│   └── arena50raio.obj
│
├── reward_functions/
│
├── worlds/
│   ├── arena.wbt
│   └── track.wbt
│
├── .gitignore
└── README.md
```

---

# Vehicle Modifications for Drifting

The original Webots BMW X5 model was modified to make drifting behaviour possible and easier to learn.

## Vehicle Parameters

| Parameter                   | Original                | Modified           |
|-----------------------------|-------------------------|--------------------|
| Mass                        | 2000 kg                 | 1100 kg            |
| Center of Mass              | 1.2975 0 0.1            | 1.0 0 -0.2         |
| Front Track Width           | 1.628 m                 | 1.8 m              |
| Rear Track Width            | 1.628 m                 | 1.5 m              |
| Wheel Base                  | 2.995 m                 | 2.99 m             |
| Time 0–100 km/h             | 7 s                     | 3 s                |
| Engine Min RPM              | 1000                    | 1500               |
| Engine Max RPM              | 4500                    | 8000               |
| Engine Function Coefficients | 34.11 0.136 -0.00001461 | 600 2 -0.0004      |
| Max Power                   | Default                 | 400000 W           |
| Max Torque                  | Default                 | 1600 Nm            |
| Max Steering Angle          | ≈ 0.5 rad               | 2.5 rad            |
| Type                        | Default (AWD)           | "propulsion" (RWD) |
| Wheel Mass                  | 30 kg                   | 50 kg              |
| Wheel Damping               | 5                       | 2                  |
---

## Suspension Changes

| Parameter | Original  | Modified |
|------------|-----------|----------|
| Front Spring Constant | undefined | 120000 N/m |
| Rear Spring Constant | undefined | 90000 N/m |
| Front Damping | undefined | 12000 N·s/m |
| Rear Damping | undefined | 3000 N·s/m |

These changes increase rear instability and weight transfer, making drift initiation easier.

---

# World Physics Configuration

The physics configuration was also modified.

## Simulation Timestep

Experiments were conducted using:

```text
basicTimeStep = 32 ms
```

and

```text
basicTimeStep = 8 ms
```

to evaluate the influence of simulation frequency on learning performance.

---

## ODE Physics Parameters

| Parameter | Default | Modified |
|------------|----------|----------|
| ERP | 0.6 | 0.2 |
| CFM | Undefined | 0.0001 |
| softCFM | Undefined | 0.0002 |

These values soften contact constraints and help produce smoother tyre sliding behaviour.

---

## Wheel Contact Material

A custom wheel material was introduced:

```text
"BmwX5Wheels"
```

and assigned to:

```text
contactMaterial
```

to reduce tyre grip and facilitate drifting on low-friction surfaces.

---

# Notes

- PPO uses a continuous action space consisting of steering and throttle.
- DQN uses a discretized version of the same action space.
- Checkpoints and logs are automatically saved during training.
- Evaluation metrics collected through custom callbacks can be found in:

```text
logs/eval_metrics/
```

---

# Code Availability

The complete implementation is available at:

```text
https://github.com/gabyru12/Introduction_robotics
```