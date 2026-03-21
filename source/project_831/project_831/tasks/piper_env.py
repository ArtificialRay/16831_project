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
        self._gripper_base_body_idx = body_names.index(self.cfg.gripper_base_name)

        # buffers
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._max_steps = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

        # arm limits as tensors on device
        self._arm_joint_limits = torch.tensor(
            self.cfg.joint_limits_rad, dtype=torch.float32, device=self.device
        )
        self._arm_lower_limits = self._arm_joint_limits[:, 0]
        self._arm_upper_limits = self._arm_joint_limits[:, 1]

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

        super()._reset_idx(env_ids)

        box_state = self.box.data.default_root_state[env_ids].clone()
        env_origins = self.scene.env_origins[env_ids]

        # print(f"[reset] num_envs_reset={env_ids.shape[0]}, env_ids={env_ids[:8].detach().cpu().tolist()}")

        # -------------------------------------------------
        # Reset box position with randomization
        # -------------------------------------------------
        n = env_ids.shape[0]
        dx = (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.box_randomize_xy[0]
        dy = (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.box_randomize_xy[1]

        box_state[:, 0] = env_origins[:, 0] + self.cfg.box_init_pos[0] + dx
        box_state[:, 1] = env_origins[:, 1] + self.cfg.box_init_pos[1] + dy
        box_state[:, 2] = env_origins[:, 2] + self.cfg.box_center_z

        box_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        box_state[:, 7:] = 0.0

        self.box.write_root_state_to_sim(box_state, env_ids=env_ids)

        # -------------------------------------------------
        # Reset robot joints
        # -------------------------------------------------
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # Set explicit arm reset pose
        joint_pos[:, self._arm_joint_ids] = self._default_arm_joint_pos[env_ids]

        # Set gripper open
        joint_pos[:, self._link7_joint_id] = self.cfg.gripper_open
        joint_pos[:, self._link8_joint_id] = -self.cfg.gripper_open

        # Zero joint velocities
        joint_vel[:] = 0.0

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # -------------------------------------------------
        # Reset target position with env-relative sampling
        # -------------------------------------------------
        max_tries = 32
        target_xy = torch.zeros((n, 2), device=self.device)
        box_xy = box_state[:, 0:2]

        valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        for _ in range(max_tries):
            tx = env_origins[:, 0] + self.cfg.target_pos[0] + (
                (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.target_randomize_xy[0]
            )
            ty = env_origins[:, 1] + self.cfg.target_pos[1] + (
                (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.target_randomize_xy[1]
            )

            proposal = torch.stack([tx, ty], dim=-1)
            dist = torch.linalg.norm(proposal - box_xy, dim=-1)

            new_valid = (dist >= self.cfg.min_goal_dist_from_box) & (
                dist <= self.cfg.max_goal_dist_from_box
            )

            write_mask = (~valid) & new_valid
            target_xy[write_mask] = proposal[write_mask]
            valid |= new_valid

            if torch.all(valid):
                break

        if not torch.all(valid):
            target_xy[~valid, 0] = env_origins[~valid, 0] + self.cfg.target_pos[0]
            target_xy[~valid, 1] = env_origins[~valid, 1] + self.cfg.target_pos[1]

        self.target_pos[env_ids, 0:2] = target_xy
        self.target_pos[env_ids, 2] = env_origins[:, 2] + self.cfg.box_center_z

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
        '''
        jaw_l_quat = self.robot.data.body_quat_w[:, self._jaw_l_body_idx, :]
        gripper_rpy = torch.stack(euler_xyz_from_quat(jaw_l_quat), dim=-1)
        gripper_pose = torch.cat([gripper_pos, gripper_rpy], dim=-1)
        '''
        gripper_base_quat = self.robot.data.body_quat_w[:, self._gripper_base_body_idx, :].clone()
        flip = gripper_base_quat[:, 0] < 0
        gripper_base_quat[flip] = -gripper_base_quat[flip]

        gripper_pose = torch.cat([gripper_pos, gripper_base_quat], dim=-1)

        # -------------------------------------------------
        # box pose
        # -------------------------------------------------
        '''
        box_quat = self.box.data.root_quat_w
        box_rpy = torch.stack(euler_xyz_from_quat(box_quat), dim=-1)
        object_pose = torch.cat([box_pos, box_rpy], dim=-1)
        '''
        box_pos = self.box.data.root_pos_w
        box_quat = self.box.data.root_quat_w.clone()
        flip = box_quat[:, 0] < 0
        box_quat[flip] = -box_quat[flip]

        object_pose = torch.cat([box_pos, box_quat], dim=-1)

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
        time_out = self.episode_length_buf >= self._max_steps - 1

        terminated = success | box_fell
        truncated = time_out

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos

        jaw_l_pos = self.robot.data.body_pos_w[:, self._jaw_l_body_idx, :]
        jaw_r_pos = self.robot.data.body_pos_w[:, self._jaw_r_body_idx, :]
        gripper_pos = 0.5 * (jaw_l_pos + jaw_r_pos)

        box_pos = self.box.data.root_pos_w
        target_pos = self.target_pos

        joint7_q = joint_pos[:, self._link7_joint_id]
        joint8_q = joint_pos[:, self._link8_joint_id]
        grip_opening = 0.5 * (joint7_q - joint8_q)
        gripper_open_dist = torch.linalg.norm(jaw_r_pos - jaw_l_pos, dim=-1)

        # -------------------------------------------------
        # Incentivize gripper reaching the box
        # -------------------------------------------------
        grip_to_box_dist = torch.linalg.norm(box_pos - gripper_pos, dim=-1)
        box_to_target_dist = torch.linalg.norm(box_pos - target_pos, dim=-1)

        grip_to_box_dist = torch.linalg.norm(box_pos - gripper_pos, dim=-1)

        r_reach_box = 1.0 - torch.tanh(8.0 * grip_to_box_dist)

        r_reach = r_reach_box

        # -------------------------------------------------
        # Incentivize gripper top-down orientation; encourage grasping from top
        # -------------------------------------------------
        gripper_base_pos = self.robot.data.body_pos_w[:, self._gripper_base_body_idx, :]

        base_to_center = gripper_base_pos - gripper_pos

        xy_err = torch.linalg.norm(base_to_center[:, :2], dim=-1)

        r_topdown_proxy = 1.0 - torch.tanh(15.0 * xy_err)
        
        # -------------------------------------------------
        # Incentivize box in between gripper; encourage gripping
        # -------------------------------------------------
        grip_center = 0.5 * (jaw_l_pos + jaw_r_pos)
        xy_dist = torch.linalg.norm((box_pos - grip_center)[:, :2], dim=-1)

        r_center = 1.0 - torch.tanh(15.0 * xy_dist)

        # -------------------------------------------------
        # Incentivize gripper at grasping height; encourage reaching top of the box
        # -------------------------------------------------
        z_target = box_pos[:, 2] + self.cfg.desired_grasp_z_offset
        z_err = torch.abs(gripper_pos[:, 2] - z_target)
        
        r_z_align = 1.0 - torch.tanh(15.0 * z_err)
        
        # -------------------------------------------------
        # Incentivize box between gripper
        # -------------------------------------------------
        v_l = jaw_l_pos - box_pos
        v_r = jaw_r_pos - box_pos

        v_l_norm = torch.nn.functional.normalize(v_l, dim=-1, eps=1e-6)
        v_r_norm = torch.nn.functional.normalize(v_r, dim=-1, eps=1e-6)

        cos_lr = torch.sum(v_l_norm * v_r_norm, dim=-1)
        r_between = 0.5 * (1.0 - cos_lr)

        # r_open = 0.5 * (1.0 + torch.tanh(20.0 * (gripper_open_dist - 0.05)))
        grip_open_norm = torch.clamp(grip_opening / self.cfg.gripper_open, 0.0, 1.0)
        r_open = grip_open_norm ** 2
        
        opening_err = torch.abs(grip_opening - self.cfg.desired_grasp_opening)
        r_close = 1.0 - torch.tanh(80.0 * opening_err)
        
        in_gripper = r_between > 0.8
        not_in_gripper = ~in_gripper

        r_open = r_open * not_in_gripper + 10 * r_close * in_gripper

        r_grasp_centering = 15 * r_between + r_open

        # -------------------------------------------------
        # Incentivize lifting the box
        # -------------------------------------------------
        lift_amount = torch.clamp(box_pos[:, 2] - self.cfg.box_center_z, min=0.0)

        r_lift = torch.clamp(lift_amount / self.cfg.desired_lift_height, max=1.0)



        jaw_l_z = jaw_l_pos[:, 2]
        jaw_r_z = jaw_r_pos[:, 2]

        table_z = self.cfg.table_top_z
        threshold = self.cfg.table_safe_threshold

        # distance above table
        dist_l = jaw_l_z - table_z
        dist_r = jaw_r_z - table_z

        violation_l = dist_l < threshold
        violation_r = dist_r < threshold

        table_violation = violation_l | violation_r

        r_table_penalty = table_violation.float()

        

        # -------------------------------------------------
        # Incentivize placing after reaching desired height
        # -------------------------------------------------
        lifted = lift_amount > self.cfg.lift_threshold

        r_place = (1.0 - torch.tanh(5.0 * box_to_target_dist)) * lifted.float()

        # -------------------------------------------------
        # Incentivize placing at target position
        # -------------------------------------------------
        success = box_to_target_dist < self.cfg.success_threshold
        r_success = success.float()

        # -------------------------------------------------
        # Small penalty for actions
        # -------------------------------------------------
        r_action_penalty = torch.sum(self._actions[:, :6] ** 2, dim=-1)

        # -------------------------------------------------
        # penalize close to joint limit
        # -------------------------------------------------
        arm_q = self.robot.data.joint_pos[:, self._arm_joint_ids]

        lower = self._arm_lower_limits.unsqueeze(0)
        upper = self._arm_upper_limits.unsqueeze(0)

        q_norm = (arm_q - lower) / (upper - lower)
        dist_to_limit = torch.minimum(q_norm, 1.0 - q_norm)

        margin = self.cfg.joint_limit_margin
        joint_limit_violation = torch.clamp(margin - dist_to_limit, min=0.0) / margin

        r_joint_limit_penalty = joint_limit_violation.sum(dim=-1)

        # -------------------------------------------------
        # Overall Reward Calculation
        # -------------------------------------------------
        reward = (
            self.cfg.reach_reward_weight * r_reach
            + self.cfg.xy_plane_align_weight * r_center
            + self.cfg.z_align_weight * r_z_align
            + self.cfg.topdown_proxy_weight * r_topdown_proxy
            + self.cfg.grasp_reward_weight * r_grasp_centering
            + self.cfg.lift_reward_weight * r_lift
            # + self.cfg.place_reward_weight * r_place
            # + self.cfg.success_reward_weight * r_success
            - self.cfg.action_penalty_weight * r_action_penalty
            - self.cfg.joint_limit_penalty_weight * r_joint_limit_penalty
            - self.cfg.table_contact_penalty_weight * r_table_penalty
        )

        return reward
