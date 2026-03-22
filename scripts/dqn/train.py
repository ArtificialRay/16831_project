"""Script to train DQN agent with Isaac Lab environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import torch
import gymnasium as gym
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a DQN agent with Isaac Lab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="PiperPickNPlace", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--log_dir", type=str, default="logs/dqn", help="Directory for logging.")
parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval.")
parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="Weights & Biases project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity (team)")
parser.add_argument("--wandb-name", type=str, default=None, help="Weights & Biases run name")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="Track experiment with Weights and Biases",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging

# Add parent directory to path to import dqn_agent
sys.path.insert(0, str(Path(__file__).parent.parent))
from dqn_agent import DQN
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import project_831.tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
#from isaaclab.utils.io import dump_yaml, dump_pickle

# import logger
logger = logging.getLogger(__name__)

@hydra_task_config(args_cli.task)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Train DQN agent."""
    
    # Set seeds for reproducibility
    torch.manual_seed(args_cli.seed)
    random.seed(args_cli.seed)
    
    # Specify directory for logging experiments
    task_name = args_cli.task
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join(args_cli.log_dir, task_name))
    log_dir_name = f"seed_{args_cli.seed}_{timestamp}"
    log_dir = os.path.join(log_root_path, log_dir_name)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    
    # Create Isaac Lab environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    
    print(f"[INFO] Number of environments: {env.unwrapped.num_envs}")
    print(f"[INFO] Observation space: {env.observation_space.shape}")
    print(f"[INFO] Action space: {env.action_space.shape}")
    
    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # # Save configuration
    config = {
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": env.unwrapped.num_envs,
        "max_iterations": args_cli.max_iterations,
        "device": args_cli.device,
        "observation_space": env.observation_space.shape[0],
        "action_space": env.action_space.shape[0],
    }
    
    # Initialize Weights & Biases
    if args_cli.track:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb
        
        project_name = args_cli.wandb_project_name or f"dqn_{task_name}"
        run_name = args_cli.wandb_name or log_dir_name
        
        wandb.init(
            project=project_name,
            entity=args_cli.wandb_entity,
            name=run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        print(f"[INFO] Tracking with W&B: {project_name}/{run_name}")
    
    # Create DQN agent
    policy = DQN(args_cli, env)
    
    # Load checkpoint if specified
    if args_cli.checkpoint is not None:
        checkpoint = torch.load(args_cli.checkpoint)
        policy.q.load_state_dict(checkpoint['q_state_dict'])
        policy.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        print(f"[INFO] Loaded checkpoint from: {args_cli.checkpoint} (step {start_step})")
    else:
        start_step = 0
    
    # Training loop
    print(f"[INFO] Starting training for {args_cli.max_iterations} iterations...")
    start_time = time.time()
    

    
    for step in range(start_step, args_cli.max_iterations):
        # Run one step
        loss, epsilon = policy.run()
        
        # Evaluation and logging
        if (step + 1) % args_cli.eval_interval == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = (step + 1 - start_step) / elapsed_time
            
            print(f"Step: {step + 1:06d} | "
                  f"Reward {policy.score:.4f} | "
                  f"TD Loss {loss:.4f} | "
                  f"Epsilon {epsilon:.4f} | "
                  f"Time: {elapsed_time:.1f}s | "
                  f"Steps/s: {steps_per_sec:.1f} | "
                  f"Buffer: {policy.replay.size()}")
            
            if args_cli.track:
                wandb.log({
                    "global_step": step + 1,
                    "Reward":policy.score,
                    "TD Loss":loss,
                    "Epsilon":epsilon,
                    "time_elapsed": elapsed_time,
                    "steps_per_second": steps_per_sec,
                    "buffer_size": policy.replay.size(),
                }, step=step + 1)
            policy.score = 0 # clear previous score
        
        # Save checkpoint
        if (step + 1) % args_cli.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, "checkpoints", f"checkpoint_{step + 1:06d}.pth")
            torch.save({
                'step': step + 1,
                'q_state_dict': policy.q.state_dict(),
                'q_target_state_dict': policy.q_target.state_dict(),
                'optimizer_state_dict': policy.optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(log_dir, "checkpoints", "final.pth")
    torch.save({
        'step': args_cli.max_iterations,
        'q_state_dict': policy.q.state_dict(),
        'q_target_state_dict': policy.q_target.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'config': config,
    }, final_checkpoint_path)
    print(f"[INFO] Saved final checkpoint: {final_checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"[INFO] Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Close environment
    env.close()
    
    if args_cli.track:
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Exception during training: {repr(e)}")
    finally:
        # Close sim app
        simulation_app.close()