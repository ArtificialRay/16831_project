from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg

import isaaclab.sim.spawners as spawners
import isaaclab.sim as sim_utils
import isaaclab.actuators as actuators
import gymnasium as gym

@configclass
class PiperSwingSceneCfg(InteractiveSceneCfg):

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.2, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),          # enable collisions
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) # static table
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.2)),
    )
    table_side = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table_Side",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 1.2, 0.40),
            collision_props=sim_utils.CollisionPropertiesCfg(),          # enable collisions
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) # static table
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.2 # Other material properties can also be set
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.15)),
    )
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawners.UsdFileCfg(
            usd_path="/home/droplab/16831/project/project_831/assets/robots/piper_description.usd",
            # Optional but nice defaults:
            # scale=(1.0, 1.0, 1.0),
            # activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.20),           # lift robot base by 20cm
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # Apply to all joints. If your arm includes grippers you want excluded,
            # we can refine the joint pattern.
            "arm": actuators.ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=400.0,
                damping=40.0,
            ),
        },
    )


@configclass
class PiperSwingEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=1)
    scene: PiperSwingSceneCfg = PiperSwingSceneCfg(num_envs=1, env_spacing=2.0)

    action_scale: float = 0.8
    
    observation_space: gym.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
    action_space: gym.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
    
    decimation: int = 10
    episode_length_s: float = 3.0

    # reward weights / thresholds
    reach_weight: float = 1.0
    lift_reward: float = 10.0
    lift_height: float = 0.62       # meters (adjust: table top + margin)
    action_penalty: float = 0.001

    # success detection stability
    success_hold_steps: int = 5     # must be above lift_height for N env steps

    jaw_left_relpath: str = "piper/link7"
    jaw_right_relpath: str = "piper/link8"
