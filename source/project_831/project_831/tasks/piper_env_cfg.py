import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg


@configclass
@configclass
class PiperSceneCfg(InteractiveSceneCfg):
    num_envs: int = 64
    env_spacing: float = 2.0

    # wall under robot, top surface at z = 0.0
    table_side = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table_Side",
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 1.20, 0.40),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.20),   # top surface = 0.0
        ),
    )

    # table in front of robot, tabletop 10 cm below robot base
    # thickness = 0.10, center z = -0.15 => tabletop z = -0.10
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 1.20, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.0, -0.15),
        ),
    )

    # box resting on table:
    # tabletop z = -0.10, half box height = 0.025 => center z = -0.075
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, 0.00, -0.075),
        ),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path="/home/droplab/16831/project/project_831/assets/robots/piper_description/piper_description.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joint1": 0.0,
                "joint2": 1.7,
                "joint3": -1.1,
                "joint4": 0.0,
                "joint5": 0.5,
                "joint6": 0.0,
                "joint7": 0.035,
                "joint8": -0.035,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                stiffness=400.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint7", "joint8"],
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )


@configclass
class PiperPickNPlaceEnvCfg(DirectRLEnvCfg):
    scene: PiperSceneCfg = PiperSceneCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=20,
        physx=PhysxCfg(),
    )

    decimation: int = 20
    episode_length_s: float = 8.0 # 48 environment steps
    is_finite_horizon: bool = True

    action_space: int = 7
    observation_space: int = 24
    state_space: int = 0

    # physical joint names
    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    gripper_joint_names = ["joint7", "joint8"]

    # body names for gripper pose computation
    jaw_left_name: str = "link7"
    jaw_right_name: str = "link8"
    gripper_base_name: str = "link5"

    # reset posture
    default_arm_joint_pos = [0.0, 1.7, -1.1, 0.0, 0.5, 0.0]

    # physical finger joint targets
    gripper_open: float = 0.035
    gripper_closed: float = 0.0

    joint_limits_rad = [
        [-2.6179,   2.6179],
        [ 0.0,      3.14],
        [-2.967,    0.0],
        [-1.745,    1.745],
        [-1.22,     1.22],
        [-2.09439,  2.09439],
    ]

    arm_action_scale: float = 0.20
    gripper_action_scale: float = 1.0

    wall_top_z: float = 0.0

    table_center = (0.40, 0.0, -0.15)
    table_size = (1.20, 1.20, 0.10)
    table_top_z: float = -0.10

    box_size: float = 0.05
    box_center_z: float = -0.075

    box_init_pos = (0.35, 0.00, -0.075)
    box_randomize_xy = (0.02, 0.02)

    target_pos = (0.45, 0.00, -0.075)
    target_randomize_xy = (0.04, 0.04)

    min_goal_dist_from_box: float = 0.08
    max_goal_dist_from_box: float = 0.15

    use_abstract_gripper_state: bool = True
    pose_obs_type: str = "quat"

    joint_pos_obs_scale: float = 1.0
    gripper_pos_obs_scale: float = 1.0
    object_pos_obs_scale: float = 1.0
    relative_pos_obs_scale: float = 1.0

    desired_grasp_z_offset: float = 0.005
    desired_grasp_opening: float = 0.025
    desired_lift_height: float = 0.05
    table_safe_threshold: float = 0.01
    joint_limit_margin: float = 0.05

    # rewards weights

    reach_reward_weight: float = 3.0
    xy_plane_align_weight: float = 4.0
    z_align_weight: float = 4.0
    topdown_proxy_weight: float = 1.0
    grasp_reward_weight: float = 4.0
    lift_reward_weight: float = 50.0
    place_reward_weight: float = 6.0
    success_reward_weight: float = 20.0
    action_penalty_weight: float = 0.005
    joint_limit_penalty_weight: float = 0.2
    table_contact_penalty_weight: float = 2.0

    success_threshold: float = 0.04
    near_box_threshold: float = 0.05
    closed_threshold: float = 0.10
    lift_threshold: float = 0.05
    box_fall_threshold: float = 0.10