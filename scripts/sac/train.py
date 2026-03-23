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
parser = argparse.ArgumentParser(description="Train a SAC agent with Isaac Lab.")
# parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="PiperPickNPlace", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sac_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=12000, help="Maximum training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--log_dir", type=str, default="logs/sac", help="Directory for logging.")
parser.add_argument("--save-frequency", type=int, default=2000, help="Checkpoint save interval.")
#parser.add_argument("--eval_interval", type=int, default=200, help="Evaluation interval.")
parser.add_argument("--num_train_steps",type=int,default=1e5,help="Total training step")
parser.add_argument("--num_seed_steps",type=int,default=100,help="step to update, cannot be too large(<=100)")
parser.add_argument("--eval-frequency",type=int,default=1000,help="frequency to evaluate agent")
parser.add_argument("--num_eval_episodes",type=int,default=10,help="number of episode to evaluate")
# parser.add_argument("--wandb-project-name", type=str, default=None, help="Weights & Biases project name")
# parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity (team)")
# parser.add_argument("--wandb-name", type=str, default=None, help="Weights & Biases run name")
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
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
from omegaconf import OmegaConf

# Add scripts/ and scripts/sac/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
sys.path.insert(0, str(Path(__file__).parent))          # scripts/sac/

from sac import SACAgent
from replay_buffer import ReplayBuffer
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import project_831.tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)

logger = logging.getLogger(__name__)


class SimpleLogger:
    """Minimal logger stub satisfying the interface expected by SACAgent internals."""
    def log(self, key, val, step): pass
    def log_histogram(self, key, val, step): pass
    def log_param(self, key, val, step): pass
    def dump(self, step): pass

def run(env,agent, replay_buffer, sac_logger, log_dir, device, args_cli, num_seed_steps,num_envs,mini_batch_size,start_step=0):
    """Train SAC agent."""
    # Initialize environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    print(f"[INFO] Starting training for {args_cli.max_iterations} steps...")
    start_time = time.time()
    episode_reward = torch.zeros(num_envs,device=device)
    episode=0
    finished = torch.zeros(num_envs,dtype=torch.bool,device=device)

    for step in range(start_step, args_cli.max_iterations):
        if finished.all():
            Train_log = f"[Train] Step:{step+1} | Episode:{episode+1} | "
            Eval_log = f"[Eval] Step:{step+1} | "
            if step>0:
                Train_log += f"Duration:{time.time() - start_time} | "
                start_time = time.time()
            if step>0 and (step + 1) % args_cli.eval_frequency == 0:
                Eval_log += f"Episode:{episode+1} | "
                evaluate(
                    env=env,
                    agent=agent,
                    args_cli=args_cli,
                    step=step,
                    num_envs=num_envs,
                    device=device,
                    eval_log = Eval_log
                    )
            Train_log+= f"Episode Reward: {episode_reward.mean().item()} | "
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"]
            agent.reset()
            finished.zero_()
            episode_reward.zero_()
            episode += 1
            print(Train_log)
            print(Eval_log)

        # Random seed phase vs. policy action
        if step < num_seed_steps:
            action = torch.stack([
                torch.tensor(env.action_space.sample(), dtype=torch.float32)
                for _ in range(num_envs)
            ]).to(device)
        else:
            with torch.no_grad():
                action = agent.act(obs, sample=True)
        # Update agent after seed phase
        if step >= num_seed_steps and replay_buffer.size() > (mini_batch_size+num_envs)//num_envs:
            agent.update(replay_buffer, sac_logger, step)

        # Environment step
        next_obs_dict, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs_dict["policy"]
        done = terminated | truncated
        # done = terminated # refer to not_dones_no_max 
        replay_buffer.push(obs, action, reward, next_obs, 1 - terminated.float())
        episode_reward += reward * (~finished).float()
        finished |= done
        obs = next_obs

        # Logging
        # if (step + 1) % args_cli.eval_frequency == 0:
        #     elapsed = time.time() - start_time
        #     avg_reward = scores.mean().float() / args_cli.eval_frequency
        #     print("[Train] "
        #         f"Step: {step+1:06d} | "
        #           f"Reward: {avg_reward:.4f} | "
        #           f"Alpha: {agent.alpha.item():.4f} | "
        #           f"Duration: {elapsed:.1f}s | "
        #           f"Steps/s: {(step + 1 - start_step) / elapsed:.1f} | "
        #           f"Buffer: {replay_buffer.size()}")
            # evaluate(
            #     env=env,
            #     agent=agent,
            #     args_cli=args_cli,
            #     step=step,
            #     num_envs=num_envs,
            #     device=device)
            #scores.zero_()

        # Checkpoint
        if (step + 1) % args_cli.save_frequency == 0:
            ckpt_path = os.path.join(log_dir, "checkpoints", f"checkpoint_{step+1:06d}.pth")
            torch.save({
                "step": step + 1,
                "actor_state_dict": agent.actor.state_dict(),
                "critic_state_dict": agent.critic.state_dict(),
                "critic_target_state_dict": agent.critic_target.state_dict(),
                "actor_optimizer": agent.actor_optimizer.state_dict(),
                "critic_optimizer": agent.critic_optimizer.state_dict(),
                "log_alpha": agent.log_alpha,
                "log_alpha_optimizer": agent.log_alpha_optimizer.state_dict(),
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    env.close()

def evaluate(env,agent,args_cli,step,num_envs,device,eval_log):
    average_episode_reward = 0
    for episode in range(args_cli.num_eval_episodes):
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        agent.reset()
        #self.video_recorder.init(enabled=(episode == 0))
        episode_reward = torch.zeros(num_envs, device=device)
        finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        while not finished.all():
            agent.train(False)
            with torch.no_grad():
                action = agent.act(obs, sample=False)
            agent.train(True)
            next_obs_dict, reward, done, _ = env.step(action)
            obs = next_obs_dict["policy"]
            #self.video_recorder.record(self.env)
            episode_reward += reward

        average_episode_reward += episode_reward.mean().item()
        #self.video_recorder.save(f'{self.step}.mp4')
    average_episode_reward /= args_cli.num_eval_episodes
    eval_log +=f"Step: {step+1:06d} | Reward: {average_episode_reward:.4f} | "
    # self.logger.log('eval/episode_reward', average_episode_reward,
    #                 self.step)
    # self.logger.dump(self.step)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train SAC agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    torch.manual_seed(args_cli.seed)
    random.seed(args_cli.seed)

    # Logging directory
    task_name = args_cli.task
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join(args_cli.log_dir, task_name))
    log_dir = os.path.join(log_root_path, f"seed_{args_cli.seed}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    num_envs   = env.unwrapped.num_envs
    device     = env_cfg.sim.device

    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}, num_envs={num_envs}")

    # SAC hyperparameters from agent_cfg YAMLd
    params = agent_cfg.get("params",{})
    mini_batch_size= params.get("batch_size", 128)
    actor_update_frequency= params.get("actor_update_frequency", 1)
    critic_target_update_freq= params.get("critic_target_update_frequency", 2)
    num_seed_steps = args_cli.num_seed_steps
    # Build hydra-style configs for actor and critic so SACAgent can instantiate them
    # transfer python dict to OmegaConf dict for hydra.utils.instantiate
    critic_cfg = params.get("critic_cfg",{})
    actor_cfg = params.get("actor_cfg",{})

    sac_logger = SimpleLogger()
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_range=params.get("action_range",[-1.0, 1.0]),
        device=device,
        critic_cfg=critic_cfg,
        actor_cfg=actor_cfg,
        discount=params.get("discount", 0.99),
        init_temperature=params.get("init_temperature", 0.1),
        alpha_lr=params.get("alpha_lr", 3e-4),
        alpha_betas=params.get("alpha_betas",[0.9, 0.999]),
        actor_lr=params.get("actor_lr", 3e-4),
        actor_betas=params.get("actor_betas",[0.9, 0.999]),
        actor_update_frequency=actor_update_frequency,
        critic_lr=params.get("critic_lr", 3e-4),
        critic_betas=params.get("critic_betas",[0.9, 0.999]),
        critic_tau=params.get("critic_tau", 0.005),
        critic_target_update_frequency=critic_target_update_freq,
        batch_size=mini_batch_size,
        learnable_temperature=params.get("learnable_temperature", True),
        num_envs=num_envs
    )

    # Load checkpoint if specified
    if args_cli.checkpoint is not None:
        ckpt = torch.load(args_cli.checkpoint)
        agent.actor.load_state_dict(ckpt["actor_state_dict"])
        agent.critic.load_state_dict(ckpt["critic_state_dict"])
        agent.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        agent.log_alpha = ckpt["log_alpha"]
        agent.log_alpha_optimizer.load_state_dict(ckpt["log_alpha_optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[INFO] Loaded checkpoint from: {args_cli.checkpoint} (step {start_step})")
    else:
        start_step = 0

    replay_buffer = ReplayBuffer(num_envs=num_envs)

    run(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        sac_logger=sac_logger,
        log_dir=log_dir,
        device=device,
        args_cli=args_cli,
        num_seed_steps=num_seed_steps,
        num_envs=num_envs,
        mini_batch_size=mini_batch_size,
        start_step=start_step
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Exception during training: {repr(e)}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
