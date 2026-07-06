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

# Running the Project

Before running any script:

1. Open **Webots**.
2. Load either:

```text
worlds/arena.wbt # checkpoints are from this world
```

or

```text
worlds/track.wbt
```

3. Press **Play** in Webots.

The Python controller will connect automatically when one of the scripts below is executed.

---

# PPO Training

Training is performed using:

```text
controllers/
└── vehicle_reinforcement_learning/
    └── ex1/
        └── vehicle_env/
            └── training_ppo.py
```

## Train a New Model

Open `training_ppo.py` and set

```python
learn_new = True
```

Then run:

```text
training_ppo.py
```

During training the script automatically:

- creates the Gymnasium environment;
- trains a PPO agent;
- periodically evaluates the policy;
- saves checkpoints every 100,000 steps;
- records TensorBoard logs;
- stores detailed evaluation metrics in JSON format.

The generated files are saved inside:

```text
logs/
├── best_model/
├── checkpoints/
├── eval_logs_results/
├── eval_metrics/
└── tensorboard_logs/
```

---

## Continue Training from a Checkpoint

To resume training, set

```python
learn_new = False
```

and modify

```python
checkpoint_path = "logs/<checkpoint_folder>/ppo_vehicle_xxxxx_steps.zip"
```

Then execute:

```text
training_ppo.py
```

Training resumes from the selected checkpoint while preserving the PPO timestep counter.

---

# DQN Training

The DQN implementation is available in

```text
training_dqn.py
```

It trains a DQN agent using a discretised version of the steering and throttle action space.

Run the script directly after opening the desired Webots world.

---

# Running a Trained Policy

Inference is performed using

```text
controllers/
└── vehicle_reinforcement_learning/
    └── ex1/
        └── vehicle_env/
            └── inference.py
```

Select the desired checkpoint by modifying

```python
model = PPO.load(...)
```

and run

```text
inference.py
```

The script:

- loads the trained PPO policy;
- executes deterministic inference;
- records all environment metrics;
- saves the recorded metrics to

```text
logs/eval_metrics/inference_metrics.json
```

These metrics can later be used by the plotting utilities.

---

# Plotting Results

Three helper scripts are provided for visualising evaluation results:

```text
plot_feature_and_lap.py
```

Plots a selected metric (e.g., drift angle, slip angle or centreline distance) for a single lap.

```text
plot_feature_all_laps.py
```

Plots the selected metric across all recorded evaluation laps.

```text
plot_track_drift.py
```

Displays the driven trajectory coloured according to the drift angle.

All plotting scripts use the JSON files stored in

```text
logs/eval_metrics/
```

---

# TensorBoard

Training statistics are automatically written to

```text
logs/tensorboard_logs/
```

Launch TensorBoard from the project root:

```bash
python -m tensorboard.main --logdir logs/tensorboard_logs
```

Then open

```text
http://localhost:6006
```

to monitor training progress.

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
basicTimeStep = 32 ms #recommended
```

and

```text
basicTimeStep = 8 ms
```

to evaluate the influence of simulation frequency on learning performance.

---

## ODE Physics Parameters

| Parameter       | Default   | Modified |
|-----------------|-----------|----------|
| ERP             | 0.6       | 0.2      |
| CFM             | Undefined | 0.0001   |
| softCFM         | Undefined | 0.0002   |
| coulombFriction | 1.0       | 0.6      |


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