# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import project_831.tasks  # noqa: F401

# 6-DOF arm joint limits (rad)
JOINT_LIMITS_RAD = torch.tensor([
    [-2.6179,   2.6179],    # joint1
    [ 0.0,      3.14],      # joint2
    [-2.967,    0.0],       # joint3
    [-1.745,    1.745],     # joint4
    [-1.22,     1.22],      # joint5
    [-2.09439,  2.09439],   # joint6
], dtype=torch.float32)

# Gripper targets (choose values that match your robot’s gripper joint(s))
# If the gripper joint is in radians:
GRIPPER_OPEN = 0.4
GRIPPER_CLOSED = 0.0

def random_action_generation(env, p_close: float = 0.5):
    """
    Returns random joint targets in radians within JOINT_LIMITS_RAD.
    Output shape: (num_envs, action_dim)
    """
    device = env.unwrapped.device
    action_dim = env.action_space.shape[0]
    num_joints = JOINT_LIMITS_RAD.shape[0]

    limits = JOINT_LIMITS_RAD.to(device)

    low = limits[:, 0]
    high = limits[:, 1]

    # Uniform sample in [0,1] then scale to [low, high]
    u = torch.rand(action_dim, device=device)

    limits = JOINT_LIMITS_RAD.to(device)
    low, high = limits[:, 0], limits[:, 1]

    # Arm: uniform in [low, high]
    u = torch.rand(num_joints, device=device)
    arm = low + (high - low) * u

    # Gripper: binary open/close
    close_mask = (torch.rand(1, device=device) < p_close)
    gripper = torch.where(
        close_mask,
        torch.tensor(GRIPPER_CLOSED, device=device),
        torch.tensor(GRIPPER_OPEN, device=device),
    )

    return torch.cat([arm, +gripper, -gripper], dim=0)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    env_steps = 0
    max_env_steps = 5_000

    next_log = 100
    action_repeat = 20
    actions = None

    ep_returns = torch.zeros((1,), device=env.unwrapped.device)
    ep_lengths = torch.zeros((1,), dtype=torch.int32, device=env.unwrapped.device)

    # Data for plot: (env_steps, mean_return_over_recent_episodes)
    xs = []
    ys = []

    completed_episode_returns = []

    obs, info = env.reset()

    while simulation_app.is_running() and env_steps < max_env_steps:
        with torch.inference_mode():

            if env_steps % action_repeat == 0:
                actions = random_action_generation(env)

            obs, reward, terminated, truncated, info = env.step(actions)
            env_steps += 1

            # print("[DEBUG] reward: ", reward)
            ep_returns += reward
            ep_lengths += 1

            done = terminated | truncated  # (num_envs,)

            if torch.any(done):
                done_ids = torch.nonzero(done).squeeze(-1)

                # Record returns for finished episodes
                completed_episode_returns.extend(ep_returns[done_ids].detach().cpu().tolist())

                # Reset trackers for those envs
                ep_returns[done_ids] = 0.0
                ep_lengths[done_ids] = 0

            # Log a point every log_every env-steps
            if env_steps >= next_log:
                if len(completed_episode_returns) > 0:
                    # mean of the most recent episodes (e.g., last 50) to smooth
                    window = completed_episode_returns[-50:]
                    mean_ret = float(np.mean(window))
                else:
                    mean_ret = float("nan")

                xs.append(env_steps)
                ys.append(mean_ret)
                print(f"[eval] steps={env_steps}  mean_return(last<=50 eps)={mean_ret:.3f}")
                next_log += 100

    # close the simulator
    env.close()

    # Plot
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Environment steps")
    plt.ylabel("Mean episodic return (recent episodes)")
    plt.title(f"Random agent performance: project831-PiperSwing-v0")
    plt.grid(True)

    out_png = "results/random_agent_return_curve.png"
    out_csv = "results/random_agent_return_curve.csv"
    plt.savefig(out_png, dpi=150)

    # Save data too (nice for proposal appendix)
    np.savetxt(out_csv, np.column_stack([xs, ys]), delimiter=",", header="env_steps,mean_return", comments="")
    print(f"Saved plot: {out_png}")
    print(f"Saved data: {out_csv}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
