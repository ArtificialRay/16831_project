import torch
import gymnasium as gym
from pxr import UsdGeom
import omni.usd
from pxr import Usd


from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject

from .piper_env_cfg import PiperSwingEnvCfg


class PiperSwingEnv(DirectRLEnv):
    cfg: PiperSwingEnvCfg

    def __init__(self, cfg: PiperSwingEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self.target: RigidObject = self.scene["box"]
        self.num_joints = self.robot.num_joints
        
        # Now that we know num_joints, set correct spaces (and keep them in cfg for printing/debugging)
        self.cfg.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.num_joints * 2,), dtype=float
        )
        self.cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=float)

        # Gymnasium env expects these properties too
        self.observation_space = self.cfg.observation_space
        self.action_space = self.cfg.action_space

        self.q_nominal = None

        self._step_count = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self._max_steps = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

        # After robot.initialize() has happened (post reset), body names exist.
        body_names = getattr(self.robot.data, "body_names", None)
        if body_names is None:
            body_names = self.robot.body_names  # fallback in some versions

        # Convert to python list of strings if needed
        body_names = [str(n) for n in body_names]

        self._jaw_l_idx = body_names.index("link7")
        self._jaw_r_idx = body_names.index("link8")

        self._jaw_left_path = f"/World/envs/env_0/Robot/{self.cfg.jaw_left_relpath}"
        self._jaw_right_path = f"/World/envs/env_0/Robot/{self.cfg.jaw_right_relpath}"


    def _reset_idx(self, env_ids):
        self.robot.reset(env_ids)
        self._step_count[env_ids] = 0
        if self.q_nominal is None:
            self.q_nominal = self.robot.data.joint_pos.clone()

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions and compute targets before physics stepping."""
        # Ensure shape is (num_envs, num_joints)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        self._actions = actions.to(self.device)

        # Compute position targets around nominal pose
        target_q = self.q_nominal + self.cfg.action_scale * self._actions
        self.robot.set_joint_position_target(target_q)

        self._step_count += 1

    # Some versions also call _apply_action(); safe to include.
    def _apply_action(self):
        """Apply action to simulation (targets already set)."""
        # No-op because set_joint_position_target already queued the target.
        return

    def _get_observations(self):
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        obs = torch.cat([q, qd], dim=-1)
        return {"policy": obs}

    def _get_rewards(self):
        # return torch.zeros((self.num_envs,), device=self.device)
        box_pos = self.target.data.root_pos_w          # (num_envs, 3)
        # grip_pos = self._get_gripper_center_w()     # (num_envs, 3)

        jaw_l = self.robot.data.body_pos_w[:, self._jaw_l_idx, :]  # (num_envs, 3)
        jaw_r = self.robot.data.body_pos_w[:, self._jaw_r_idx, :]  # (num_envs, 3)
        grip_pos = 0.5 * (jaw_l + jaw_r)

        dist = torch.linalg.norm(grip_pos - box_pos, dim=-1)
        r_reach = -self.cfg.reach_weight * dist

        return r_reach

    def _get_dones(self):
        terminated = torch.zeros((self.num_envs,), device=self.device)
        truncated = self._step_count >= self._max_steps
        return terminated, truncated

    '''
    def _step_impl(self, actions: torch.Tensor):
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        target_q = self.q_nominal + self.cfg.action_scale * actions
        self.robot.set_joint_position_target(target_q)
    '''