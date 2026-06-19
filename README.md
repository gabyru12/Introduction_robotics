# Autonomous Drifting Using Reinforcement Learning in Webots
 
Project developed for **Introduction to Intelligent Robotics (CC3046)**.
 
Authors:
- André Amaral
- Gabriel Oliveira
- José Sousa
 
## Compatible Operating Systems
 
The project was developed and tested on:
 
- Windows 11
- Webots R2025a
 
Compatibility with Linux or macOS has not been tested.
 
---
 
## Required Software
 
Before running the project, install:
 
### 1. Webots
 
Install Webots R2025a from:
 
Cyberbotics: Robotics simulation with Webots
 
Recommended installation path:
 
```text
C:\Program Files\Webots\
```
 
### 2. Python
 
Install Python 3.14.
 
### 3. PyCharm
 
Install JetBrains PyCharm Community Edition or Professional Edition.
 
---
 
## Python Dependencies
 
Install the following Python packages:
 
```bash
pip install gymnasium
pip install stable-baselines3
```
 
or
 
```bash
pip install gymnasium stable-baselines3
```
 
---
 
## Project Setup
 
### Step 1 — Extract the Project
 
Extract the ZIP file to any local directory.
 
Example:
 
```text
C:\autonomous_drifting\
```
 
---
 
### Step 2 — Open the Project
 
Open the extracted folder in PyCharm.
 
---
 
### Step 3 — Create a Python Interpreter
 
Create a new local Python interpreter:
 
1. Open PyCharm.
2. Click the interpreter selector at the bottom-right corner.
3. Select:
 
```text
Add New Interpreter
```
 
4. Choose:
 
```text
Local Interpreter
```
 
5. Select Python 3.14.
6. Apply the changes.
 
---
 
### Step 4 — Configure Webots Libraries
 
In PyCharm:
 
```text
File
→ Settings
→ Project Structure
→ Add Content Root
```
 
Add the following folder:
 
```text
C:\Program Files\Webots\lib\controller\python
```
 
This is required so PyCharm can detect:
 
```python
from controller import Robot
from vehicle import Driver
```
 
and other Webots APIs.
 
---
 
## Running the Project
 
### Step 1 — Open Webots
 
Launch Webots.
 
---
 
### Step 2 — Load a World
 
Navigate to:
 
```text
worlds/
```
 
and open one of the available environments:
 
```text
arena.wbt
```
 
or
 
```text
track.wbt
```
 
---
 
## PPO Training
 
Open:
 
```text
training_ppo.py
```
 
To train a new model:
 
```python
learn_new = True
```
 
Then run:
 
```text
training_ppo.py
```
 
The script will:
 
- Launch the Gymnasium environment.
- Start PPO training.
- Generate TensorBoard logs.
- Save checkpoints periodically.
- Perform evaluation episodes during training.
 
---
 
## Continue PPO Training
 
To continue training a checkpoint:
 
Set:
 
```python
learn_new = False
```
 
and specify the checkpoint path:
 
```python
checkpoint_path = "logs/.../ppo_vehicle_xxxxx_steps.zip"
```
 
Then run:
 
```text
training_ppo.py
```
 
again.
 
---
 
## DQN Training
 
Open:
 
```text
training_dqn.py
```
 
and run it.
 
This script:
 
- Converts the continuous steering/throttle space into discrete actions.
- Trains a DQN agent.
- Generates TensorBoard logs.
- Stores evaluation results.
 
---
 
## Running a Trained Agent
 
Open:
 
```text
inference.py
```
 
Modify:
 
```python
model = PPO.load(...)
```
 
to the desired checkpoint.
 
Example:
 
```python
model = PPO.load(
    "logs/drift_open_space_6h_30_60_degrees_checkpoints/ppo_vehicle_2500000_steps.zip"
)
```
 
Run:
 
```text
inference.py
```
 
The vehicle will execute the learned policy inside Webots.
 
---
 
## Visualizing Training Statistics
 
TensorBoard logs are generated inside:
 
```text
logs/tensorboard_logs
```
 
Open a terminal in the project root and run:
 
```bash
python -m tensorboard.main --logdir "./logs/tensorboard_logs"
```
 
Then open:
 
```text
http://localhost:6006
```
 
in your browser.
 
---
 
## Project Structure
 
```text
project/
│
├── controllers/
    ├── test_folder_keyboard_control
│   └── vehicle_reinforcement_learning/
│       └── ex1/
│           ├── vehicle_env/__init__.py
│           ├── training_ppo.py
│           ├── training_dqn.py
│           └── inference.py
│
├── protos/           
│
├── worlds/
│   ├── arena.wbt
│   └── track.wbt
│
├── logs/
│
└── README.md
```
 
---
 
## Notes
 
- The Arena environment was primarily used for drift-learning experiments.
- The Track environment was used for reward-shaping experiments and navigation tests.
- PPO uses a continuous action space composed of steering and throttle.
- DQN uses a discretized version of the same action space.
- Trained models and checkpoints are automatically saved inside the logs directory.
Cyberbotics: Robotics simulation with Webots
Cyberbotics - Robotics simulation with Webots
 