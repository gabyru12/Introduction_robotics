"""
my_controller.py — Main entry point for the autonomous vehicle.

What this file does:
    1. Creates the Webots Driver (BMW X5 PROTO).
    2. Instantiates the sensor manager.
    3. Picks a policy (rule-based for now; RL agents later).
    4. Runs the simulation loop: read sensors → act → command vehicle.

This file is deliberately thin. All real logic lives in sensors.py
(reading) and in rule_based.py / future agents (deciding).

The policy is selected by a constant at the top — change POLICY to swap.
This is the kind of thing that will be a CLI flag once we have multiple
agents trained.
"""
from __future__ import annotations

import os
import sys
import traceback

_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller_crash.log")

# Webots modifies the process PATH before launching the controller, adding
# its own lib directories first. On Windows this causes numpy's C extensions
# to resolve against Webots' DLLs (e.g. a bundled libopenblas) instead of
# the conda env's versions, resulting in a silent segfault.
# Fix: explicitly register the conda env's DLL directory with the Windows
# loader via os.add_dll_directory() before any C extension is imported.
_conda_lib_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
if os.path.isdir(_conda_lib_bin):
    if hasattr(os, "add_dll_directory"):   # Python 3.8+ Windows only
        os.add_dll_directory(_conda_lib_bin)
    # Also prepend to PATH as a belt-and-braces fallback.
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")

def _excepthook(exc_type, exc_value, exc_tb):
    with open(_LOG, "w") as f:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()

sys.excepthook = _excepthook

import numpy as np
from vehicle import Driver
from sensors import VehicleSensors
from rule_based import (
    RuleBasedAgent, RuleBasedConfig,
    OvalRaceController, OvalRaceConfig,
    DriftDemoController, DriftDemoConfig,
    KeyboardController, KeyboardConfig,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Which policy to run. Options: "rule_based", "oval_race", "dqn", "ppo" (future).
POLICY = "keyboard"

# Maximum physical steering angle of the BMW X5 PROTO (radians).
# Webots' BmwX5.proto declares a maxSteeringAngle. We mirror it here so
# our normalised steering in [-1, 1] maps to real radians.
MAX_STEER_RAD = 0.8

# Target speed that corresponds to throttle=1.0 (km/h).
# The rule-based baseline uses setCruisingSpeed(throttle * CRUISING_SPEED_KMH).
# For RL with direct torque control, replace with setThrottle() + setGear().
CRUISING_SPEED_KMH = 60.0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def build_agent(name: str, keyboard=None):
    """Factory: return an agent that implements .act(obs) -> (steer, throttle, brake)."""
    if name == "rule_based":
        return RuleBasedAgent(RuleBasedConfig())
    if name == "oval_race":
        return OvalRaceController(OvalRaceConfig())
    if name == "drift_demo":
        return DriftDemoController(DriftDemoConfig())
    if name == "keyboard":
        if keyboard is None:
            raise ValueError("POLICY='keyboard' requires a keyboard handle from driver.getKeyboard().")
        return KeyboardController(keyboard, KeyboardConfig())
    # Placeholders for the RL phase:
    # if name == "dqn":
    #     from agents.dqn_agent import DQNAgent
    #     return DQNAgent.load("checkpoints/dqn_latest.zip")
    # if name == "ppo":
    #     from agents.ppo_agent import PPOAgent
    #     return PPOAgent.load("checkpoints/ppo_latest.zip")
    raise ValueError(f"Unknown policy: {name!r}")


def apply_action(driver: Driver, steering: float, throttle: float, brake: float, speed_ms: float = 0.0) -> None:
    """Send a normalised action to the Webots Driver.

    All three inputs are normalised:
        steering: [-1, 1]  (left negative, right positive)
        throttle: [ 0, 1]  (direct torque input via setThrottle)
        brake:    [ 0, 1]

    Uses setThrottle() + manual gear management so the internal PID of
    setCruisingSpeed() does not fight against the oversteer needed for drift.
    Gear shifts are triggered by RPM thresholds read from the Driver.
    """
    steering = max(-1.0, min(1.0, steering))
    throttle = max(0.0, min(1.0, throttle))
    brake    = max(0.0, min(1.0, brake))

    prev_steer = getattr(apply_action, "_prev_steer", 0.0)
    alpha = 0.1
    steering = prev_steer + alpha * (steering - prev_steer)
    apply_action._prev_steer = steering

    driver.setSteeringAngle(steering * MAX_STEER_RAD)
    
    if brake > 0.1:
        driver.setThrottle(0.0)
        driver.setBrakeIntensity(brake)
    else:
        driver.setBrakeIntensity(0.0)
        driver.setThrottle(throttle)

        current_gear = driver.getGear()

        gear_ratios = [0.0, 12.5, 8.0, 5.35, 4.3, 4.0]
        wheel_radius = 0.36
        if current_gear > 0 and current_gear < len(gear_ratios):
            rpm = (speed_ms / wheel_radius) * gear_ratios[current_gear] * 60.0 / 6.2832
        else:
            rpm = 1000.0

        # Cooldown entre mudanças: 0.5s mínimo entre transições para evitar hunting.
        import time as _t
        last_shift = getattr(apply_action, "_last_shift_t", 0.0)
        can_shift = (_t.time() - last_shift) > 0.5

        if current_gear == 0:
            driver.setGear(1)
            apply_action._last_shift_t = _t.time()
        elif can_shift:
            # Histerese: gap largo entre shift-up (5000) e shift-down (1500).
            if rpm > 5000 and current_gear < 5:
                driver.setGear(current_gear + 1)
                apply_action._last_shift_t = _t.time()
                prev = getattr(apply_action, "_prev_gear", -1)
                if current_gear + 1 != prev:
                    print(f"[gear] UP to {current_gear + 1} (rpm={rpm:.0f}, speed_ms={speed_ms:.1f})", flush=True)
                    apply_action._prev_gear = current_gear + 1
            elif rpm < 1500 and current_gear > 1:
                driver.setGear(current_gear - 1)
                apply_action._last_shift_t = _t.time()
                prev = getattr(apply_action, "_prev_gear", -1)
                if current_gear - 1 != prev:
                    print(f"[gear] DOWN to {current_gear - 1} (rpm={rpm:.0f}, speed_ms={speed_ms:.1f})", flush=True)
                    apply_action._prev_gear = current_gear - 1


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------


def main() -> None:
    driver = Driver()
    time_step = int(driver.getBasicTimeStep())

    sensors = VehicleSensors(driver, time_step)

    keyboard = None
    if POLICY == "keyboard":
        keyboard = driver.getKeyboard()
        keyboard.enable(time_step)

    agent = build_agent(POLICY, keyboard=keyboard)
    agent.reset()

    print(f"[main] Policy: {POLICY}")
    print(f"[main] Time step: {time_step} ms")
    print(f"[main] Lidar: res={sensors.lidar_horizontal_resolution} "
          f"fov={sensors.lidar_fov:.3f} max_range={sensors.lidar_max_range:.1f}")

    if POLICY == "keyboard":
        print("[keyboard]  UP/W=throttle  DOWN/S=brake  LEFT/A=steer_left  RIGHT/D=steer_right  SPACE=handbrake")
        print("[keyboard]  Teclas simultâneas suportadas. Handbrake > brake > throttle.", flush=True)

    step_count = 0

    while driver.step() != -1:
        current_time = driver.getTime()
        obs = sensors.read(current_time)

        steering, throttle, brake = agent.act(obs)
        apply_action(driver, steering, throttle, brake, speed_ms=abs(obs.forward_speed))

        if step_count % max(1, int(1000 / time_step)) == 0:
            print(
                f"[t={current_time:6.2f}] "
                f"speed={obs.forward_speed:+6.2f}  "
                f"drift={obs.drift_angle:+5.3f}  "
                f"steer={steering:+5.2f} thr={throttle:4.2f} brk={brake:4.2f}"
            )
        step_count += 1


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        with open(_LOG, "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise