# Autonomous Vehicle B — Setup Guide

Project for **Introduction to Intelligent Robotics (CC3046)**, FCUP.
Group: André Amaral, Gabriel Oliveira, José Sousa.

This guide explains how to set up the development environment so that the controller code runs correctly with our Webots simulation. Follow the steps in order. If something fails, stop and check the troubleshooting section at the bottom.

## What you need before starting

1. **Webots R2025a** installed at the default path `C:\Program Files\Webots\`. Download from [cyberbotics.com](https://cyberbotics.com/).
2. **Miniconda** or **Anaconda** installed. Download Miniconda from [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html).
3. **VS Code** installed. Download from [code.visualstudio.com](https://code.visualstudio.com/).
4. The project folder on your machine (whether from OneDrive, zip, or `git clone`).

All instructions assume Windows. If anyone moves to macOS or Linux later, the paths and PowerShell commands will need adjusting.

## Step 1 — Create the Conda environment

Open **Anaconda Prompt** (search "Anaconda Prompt" in the Start menu). Navigate to the project root folder:

```
cd "C:\path\to\introduction_robotics"
```

Create the environment from the project file:

```
conda env create -f environment.yml
```

This installs Python 3.11 and the required dependencies. It takes a few minutes.

Activate it to confirm:

```
conda activate intro-robotics-vb
python --version
```

You should see `Python 3.11.x`.

Find the full path to this environment's Python — you will need it for the next steps:

```
where.exe python
```

Copy the first line of output. It should look like:

```
C:\Users\<your-user>\miniconda3\envs\intro-robotics-vb\python.exe
```

Save this path somewhere; you will paste it twice in the next steps.

## Step 2 — Point Webots to the correct Python

Webots launches Python itself when you run the simulation. By default, it uses whatever `python` is on your system PATH, which is *not* our conda environment. Fix it:

1. Open Webots.
2. Go to `Tools → Preferences → General`.
3. Find the field **Python command**.
4. Replace the contents with the full path from Step 1, e.g.:
   ```
   C:\Users\<your-user>\miniconda3\envs\intro-robotics-vb\python.exe
   ```
5. Click OK / Apply.
6. Close Webots completely.

## Step 3 — Configure VS Code

Open the project folder in VS Code: `File → Open Folder` → select the `introduction_robotics` folder (the one containing `controllers/`, `worlds/`, etc.).

### Install the required extensions

Open the Extensions panel (Ctrl+Shift+X) and install **only**:

- **Python** (Microsoft) — id `ms-python.python`
- **Pylance** (Microsoft) — id `ms-python.vscode-pylance`

### Configure local paths

The repository contains a `.vscode/settings.template.json` file. Each developer must create their own `.vscode/settings.json` based on it, since it contains user-specific paths.

Copy the template:

```
copy .vscode\settings.template.json .vscode\settings.json
```

Then open `.vscode/settings.json` and replace `<YOUR-USER>` with your Windows username everywhere it appears.

### Select the interpreter

1. Press Ctrl+Shift+P.
2. Type `Python: Select Interpreter`.
3. Choose `Python 3.11.x ('intro-robotics-vb': conda)`.

The bottom-right of the VS Code window should now show `Python 3.11.x ('intro-robotics-vb': conda)`.

## Step 4 — Verify everything works

Create a temporary test file in the project root called `test_setup.py`:

```python
from controller import Robot
from vehicle import Driver

r: Robot
d: Driver
print("Imports OK")
```

**Do not run this file.** Just check that:

1. The lines `from controller import Robot` and `from vehicle import Driver` are **not** underlined in red.
2. If you type `r.` on a new line, autocomplete suggests methods like `getDevice`, `step`, `getName`.

If both pass, you can delete `test_setup.py` and you are ready to work.

To test the full pipeline:

1. Open Webots.
2. Open the world file: `File → Open World → worlds\vehicle_testing.wbt`.
3. Click the play button (▶).
4. The Webots console at the bottom should print messages from the controller without Python import errors.

## Project structure

```
introduction_robotics/
├── .vscode/
│   ├── settings.template.json   # template — copy to settings.json and edit
│   └── settings.json            # local config — do not share, contains your paths
├── controllers/
│   └── my_controller/
│       ├── my_controller.py     # entry point: simulation loop, policy selection
│       ├── sensors.py           # sensor manager and Observation dataclass
│       └── rule_based.py        # rule-based baseline agent
├── libraries/
├── plugins/
├── protos/
├── worlds/
│   └── vehicle_testing.wbt      # main simulation world
├── environment.yml              # conda environment definition
├── .gitignore                   # for future Git use
└── README.md                    # this file
```

## Workflow

1. Edit `.py` files in VS Code.
2. Save (Ctrl+S).
3. Alt-Tab to Webots.
4. Press **Ctrl+Shift+T** in Webots to reload the world (this forces Webots to re-read the controller from disk).
5. Press play (▶).

If you only press play without reloading, Webots may keep the previous version of the controller cached.

## Troubleshooting

**Pylance says `Unresolved import: controller`** — your interpreter is not pointing at the conda environment, or `.vscode/settings.json` was not copied/edited. Repeat Step 3.

**Webots console says `ModuleNotFoundError: No module named 'controller'`** — the Python command in Webots Preferences is wrong, or the conda environment doesn't exist on this machine. Repeat Steps 1 and 2.

**Webots console says `ModuleNotFoundError: No module named 'numpy'`** — the conda environment exists but the Python command in Webots is pointing at the wrong Python (probably the system one). Verify the path in `Tools → Preferences → Python command`.

**Code edits don't seem to take effect in the simulation** — press Ctrl+Shift+T in Webots before pressing play, to force a world reload.

**Project sits inside OneDrive and behaves oddly** (random file locks, slow saves) — move the project out of OneDrive to a local path like `C:\dev\introduction_robotics\`. OneDrive can interfere with Webots' file watching.

## Adding dependencies later

When we start the RL phase we will need additional packages (PyTorch, stable-baselines3, gymnasium). Whoever adds a dependency must:

1. Update `environment.yml`.
2. Tell the others on the group chat so they can update their environment with:
   ```
   conda env update -f environment.yml --prune
   ```

Do not install packages ad-hoc with `pip install` without updating the file. It will work on your machine and break on everyone else's.
