# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an Isaac Lab environment with a random-action agent."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--max_env_steps",
    type=int,
    default=5000,
    help="Maximum number of environment steps to run.",
)
parser.add_argument(
    "--log_every",
    type=int,
    default=100,
    help="Print stats every N environment steps.",
)
parser.add_argument(
    "--gripper_close_prob",
    type=float,
    default=0.5,
    help="Probability of issuing a close command on the gripper action dimension.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -----------------------------------------------------------------------------
# Launch sim
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after app launch
# -----------------------------------------------------------------------------
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import project_831.tasks  # noqa: F401


def random_action_generation(env, p_close: float = 0.5) -> torch.Tensor:
    """Generate random normalized actions for the current env.

    Action layout:
        a[:, 0:6] -> arm delta-joint commands in [-1, 1]
        a[:, 6]   -> abstract gripper command
                     <= 0 -> close
                     >  0 -> open

    Returns:
        Tensor of shape (num_envs, 7).
    """
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    action_dim = env.action_space.shape[0]

    if action_dim != 7:
        raise RuntimeError(f"Expected action_dim=7, but got {action_dim}.")

    # Random arm commands in [-1, 1]
    arm_actions = 2.0 * torch.rand((num_envs, 6), device=device) - 1.0

    # Binary-ish gripper command:
    #   -1.0 => close
    #   +1.0 => open
    close_mask = torch.rand((num_envs, 1), device=device) < p_close
    gripper_actions = torch.where(
        close_mask,
        -torch.ones((num_envs, 1), device=device),
        torch.ones((num_envs, 1), device=device),
    )

    actions = torch.cat([arm_actions, gripper_actions], dim=-1)
    return actions


def main():
    """Random-action agent for Isaac Lab environment."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space:      {env.action_space}")

    obs, info = env.reset()

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    env_steps = 0
    next_log = args_cli.log_every

    ep_returns = torch.zeros((num_envs,), dtype=torch.float32, device=device)
    ep_lengths = torch.zeros((num_envs,), dtype=torch.int32, device=device)

    completed_episode_returns = []
    completed_episode_lengths = []

    action_repeat = 4

    while simulation_app.is_running() and env_steps < args_cli.max_env_steps:
        with torch.inference_mode():

            if env_steps % action_repeat == 0:
                actions = random_action_generation(env, p_close=args_cli.gripper_close_prob)

            obs, reward, terminated, truncated, info = env.step(actions)
            env_steps += 1

            ep_returns += reward
            ep_lengths += 1

            done = terminated | truncated
            if torch.any(done):
                done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)

                completed_episode_returns.extend(ep_returns[done_ids].detach().cpu().tolist())
                completed_episode_lengths.extend(ep_lengths[done_ids].detach().cpu().tolist())

                ep_returns[done_ids] = 0.0
                ep_lengths[done_ids] = 0

            if env_steps >= next_log:
                if completed_episode_returns:
                    recent_returns = completed_episode_returns[-50:]
                    recent_lengths = completed_episode_lengths[-50:]
                    mean_ret = float(np.mean(recent_returns))
                    mean_len = float(np.mean(recent_lengths))
                    num_done = len(completed_episode_returns)
                    print(
                        f"[eval] steps={env_steps:5d} | "
                        f"episodes={num_done:4d} | "
                        f"mean_return(last<=50)={mean_ret:8.3f} | "
                        f"mean_len(last<=50)={mean_len:6.2f}"
                    )
                else:
                    print(
                        f"[eval] steps={env_steps:5d} | "
                        f"episodes=0 | "
                        f"mean_return(last<=50)=nan | "
                        f"mean_len(last<=50)=nan"
                    )
                next_log += args_cli.log_every

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
