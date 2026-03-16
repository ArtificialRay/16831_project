from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import euler_xyz_from_quat

from .piper_env_cfg import PiperPickNPlaceEnvCfg


class PiperPickNPlaceEnv(DirectRLEnv):
    cfg: PiperPickNPlaceEnvCfg

    def __init__(self, cfg: PiperPickNPlaceEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self.box: RigidObject = self.scene["box"]

        self.num_joints = self.robot.num_joints

        # --- joint/body name lookup ---
        joint_names = getattr(self.robot.data, "joint_names", None)
        if joint_names is None:
            joint_names = self.robot.joint_names
        joint_names = [str(n) for n in joint_names]

        body_names = getattr(self.robot.data, "body_names", None)
        if body_names is None:
            body_names = self.robot.body_names
        body_names = [str(n) for n in body_names]

        # arm joints
        self._arm_joint_ids = [joint_names.index(n) for n in self.cfg.arm_joint_names]

        # gripper joints (physical)
        self._gripper_joint_ids = [joint_names.index(n) for n in self.cfg.gripper_joint_names]
        self._link7_joint_id = joint_names.index(self.cfg.gripper_joint_names[0])
        self._link8_joint_id = joint_names.index(self.cfg.gripper_joint_names[1])

        # gripper body ids for pose computation
        self._jaw_l_body_idx = body_names.index(self.cfg.jaw_left_name)
        self._jaw_r_body_idx = body_names.index(self.cfg.jaw_right_name)

        # buffers
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._step_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._max_steps = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

        # arm limits as tensors on device
        self._arm_lower_limits = self.cfg.joint_limits_rad[:, 0].to(self.device)
        self._arm_upper_limits = self.cfg.joint_limits_rad[:, 1].to(self.device)

        # default joint targets
        self._default_arm_joint_pos = torch.tensor(
            self.cfg.default_arm_joint_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # keep a target-place position per env for later use
        self.target_pos = torch.tensor(
            self.cfg.target_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # gym spaces
        self.observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.cfg.observation_space,),
            dtype=float,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.action_space,),
            dtype=float,
        )

        print("joint names:", joint_names)
        print("body names:", body_names)
        print("num envs:", self.num_envs)
        print("box root positions shape:", self.box.data.root_pos_w.shape)


    # --------------------------------------------------------------------- #
    # reset
    # --------------------------------------------------------------------- #
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return

        # Reset internal asset buffers/state
        self.robot.reset(env_ids)
        self.box.reset(env_ids)

        self._step_count[env_ids] = 0

        # -------------------------------------------------
        # Reset robot joints:
        #   arm -> fixed init pose
        #   gripper -> open
        # -------------------------------------------------
        current_q = self.robot.data.joint_pos[env_ids].clone()
        current_qd = torch.zeros_like(current_q)

        # set 6 arm joints
        current_q[:, self._arm_joint_ids] = self._default_arm_joint_pos[env_ids]

        # set gripper open
        current_q[:, self._link7_joint_id] = self.cfg.gripper_open
        current_q[:, self._link8_joint_id] = -self.cfg.gripper_open

        self.robot.write_joint_state_to_sim(current_q, current_qd, env_ids=env_ids)
        self.robot.set_joint_position_target(current_q, env_ids=env_ids)

        # -------------------------------------------------
        # Reset box pose on table with XY randomization
        # -------------------------------------------------
        box_state = self.box.data.default_root_state[env_ids].clone()

        n = len(env_ids)
        dx = (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.box_randomize_xy[0]
        dy = (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.box_randomize_xy[1]

        box_state[:, 0] = self.cfg.box_init_pos[0] + dx
        box_state[:, 1] = self.cfg.box_init_pos[1] + dy
        box_state[:, 2] = self.cfg.box_center_z

        # upright orientation, zero velocities
        box_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        box_state[:, 7:] = 0.0

        self.box.write_root_state_to_sim(box_state, env_ids=env_ids)

        # -------------------------------------------------
        # Reset target position on table with rejection sampling
        # so it is not too close / too far from the box
        # -------------------------------------------------
        max_tries = 32
        target_xy = torch.zeros((n, 2), device=self.device)

        box_xy = box_state[:, 0:2]

        valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        for _ in range(max_tries):
            tx = self.cfg.target_pos[0] + (
                (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.target_randomize_xy[0]
            )
            ty = self.cfg.target_pos[1] + (
                (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.target_randomize_xy[1]
            )

            proposal = torch.stack([tx, ty], dim=-1)
            dist = torch.linalg.norm(proposal - box_xy, dim=-1)

            new_valid = (dist >= self.cfg.min_goal_dist_from_box) & (
                dist <= self.cfg.max_goal_dist_from_box
            )

            write_mask = (~valid) & new_valid
            target_xy[write_mask] = proposal[write_mask]
            valid = valid | new_valid

            if torch.all(valid):
                break

        # fallback for any remaining invalid samples
        if not torch.all(valid):
            fallback = torch.tensor(self.cfg.target_pos[:2], device=self.device)
            target_xy[~valid] = fallback

        self.target_pos[env_ids, 0:2] = target_xy
        self.target_pos[env_ids, 2] = self.cfg.box_center_z

    # --------------------------------------------------------------------- #
    # action processing
    # --------------------------------------------------------------------- #
    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        actions = torch.clamp(actions, -1.0, 1.0).to(self.device)
        self._actions = actions

        # current joint state
        current_q = self.robot.data.joint_pos.clone()
        target_q = current_q.clone()

        # -------------------------------------------------
        # arm: delta joint position control
        # -------------------------------------------------
        arm_q = current_q[:, self._arm_joint_ids]
        arm_delta = self.cfg.arm_action_scale * actions[:, :6]
        arm_target = arm_q + arm_delta
        arm_target = torch.clamp(arm_target, self._arm_lower_limits, self._arm_upper_limits)

        target_q[:, self._arm_joint_ids] = arm_target

        # -------------------------------------------------
        # gripper: one abstract scalar mapped to symmetric joints
        # > 0 => open
        # <= 0 => close
        # -------------------------------------------------
        grip_open_cmd = actions[:, 6] > 0.0
        grip_val = torch.where(
            grip_open_cmd,
            torch.full((self.num_envs,), self.cfg.gripper_open, device=self.device),
            torch.full((self.num_envs,), self.cfg.gripper_closed, device=self.device),
        )

        target_q[:, self._link7_joint_id] = grip_val
        target_q[:, self._link8_joint_id] = -grip_val

        self.robot.set_joint_position_target(target_q)

        self._step_count += 1

    def _apply_action(self):
        # no-op because targets are already sent in _pre_physics_step()
        return

    # --------------------------------------------------------------------- #
    # observations
    # --------------------------------------------------------------------- #
    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos

        # 6 arm joints
        arm_q = joint_pos[:, self._arm_joint_ids]

        # 1D abstract gripper state
        # open  = (+0.4, -0.4) -> 0.4
        # close = (0.0, 0.0)   -> 0.0
        link7_q = joint_pos[:, self._link7_joint_id]
        link8_q = joint_pos[:, self._link8_joint_id]
        grip_opening = 0.5 * (link7_q - link8_q)
        grip_opening = grip_opening.unsqueeze(-1)

        joint_state_abstract = torch.cat([arm_q, grip_opening], dim=-1)

        # -------------------------------------------------
        # gripper pose from midpoint of link7/link8 bodies
        # -------------------------------------------------
        jaw_l_pos = self.robot.data.body_pos_w[:, self._jaw_l_body_idx, :]
        jaw_r_pos = self.robot.data.body_pos_w[:, self._jaw_r_body_idx, :]
        gripper_pos = 0.5 * (jaw_l_pos + jaw_r_pos)

        # use left jaw orientation as proxy for gripper orientation
        jaw_l_quat = self.robot.data.body_quat_w[:, self._jaw_l_body_idx, :]
        gripper_rpy = torch.stack(euler_xyz_from_quat(jaw_l_quat), dim=-1)

        gripper_pose = torch.cat([gripper_pos, gripper_rpy], dim=-1)

        # -------------------------------------------------
        # box pose
        # -------------------------------------------------
        box_pos = self.box.data.root_pos_w
        box_quat = self.box.data.root_quat_w
        box_rpy = torch.stack(euler_xyz_from_quat(box_quat), dim=-1)

        object_pose = torch.cat([box_pos, box_rpy], dim=-1)

        # relative position: object - gripper
        rel_pos = box_pos - gripper_pos

        obs = torch.cat(
            [
                joint_state_abstract * self.cfg.joint_pos_obs_scale,
                gripper_pose,
                object_pose,
                rel_pos * self.cfg.relative_pos_obs_scale,
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        box_pos = self.box.data.root_pos_w

        # success: object is at target position
        goal_dist = torch.linalg.norm(box_pos - self.target_pos, dim=-1)
        success = goal_dist < self.cfg.success_threshold

        # failure: object fell too low (off table / unstable)
        box_fell = box_pos[:, 2] < (self.cfg.table_top_z - self.cfg.box_fall_threshold)

        # timeout
        time_out = self._step_count >= self._max_steps

        terminated = success | box_fell
        truncated = time_out

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos

        # -----------------------------
        # gripper state / pose
        # -----------------------------
        jaw_l_pos = self.robot.data.body_pos_w[:, self._jaw_l_body_idx, :]
        jaw_r_pos = self.robot.data.body_pos_w[:, self._jaw_r_body_idx, :]
        gripper_pos = 0.5 * (jaw_l_pos + jaw_r_pos)

        box_pos = self.box.data.root_pos_w
        target_pos = self.target_pos

        # abstract gripper opening
        joint7_q = joint_pos[:, self._link7_joint_id]
        joint8_q = joint_pos[:, self._link8_joint_id]
        grip_opening = 0.5 * (joint7_q - joint8_q)

        # -----------------------------
        # distances
        # -----------------------------
        grip_to_box_dist = torch.linalg.norm(box_pos - gripper_pos, dim=-1)
        box_to_target_dist = torch.linalg.norm(box_pos - target_pos, dim=-1)

        # -----------------------------
        # terms
        # -----------------------------
        # 1) reach reward
        r_reach = 1.0 - torch.tanh(5.0 * grip_to_box_dist)

        # 2) close-gripper bonus when near box
        near_box = grip_to_box_dist < 0.05
        gripper_closed = grip_opening < 0.1
        r_grasp_intent = (near_box & gripper_closed).float()

        # 3) lift reward
        lift_amount = torch.clamp(box_pos[:, 2] - self.cfg.box_center_z, min=0.0)
        r_lift = lift_amount

        # 4) place reward, stronger once box is lifted
        lifted = lift_amount > 0.03
        r_place = (1.0 - torch.tanh(5.0 * box_to_target_dist)) * lifted.float()

        # 5) success bonus
        success = box_to_target_dist < 0.04
        r_success = success.float()

        # 6) action penalty
        r_action_penalty = torch.sum(self._actions[:, :6] ** 2, dim=-1)

        # -----------------------------
        # weighted sum
        # -----------------------------
        reward = (
            1.5 * r_reach
            + 1.0 * r_grasp_intent
            + 4.0 * r_lift
            + 6.0 * r_place
            + 20.0 * r_success
            - 0.01 * r_action_penalty
        )

        return reward
