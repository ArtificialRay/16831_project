# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

from . import agents

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

import gymnasium as gym

from .piper_env import PiperPickNPlaceEnv
from .piper_env_cfg import PiperPickNPlaceEnvCfg

gym.register(
    id="PiperPickNPlace",
    entry_point="project_831.tasks.piper_env:PiperPickNPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "project_831.tasks.piper_env_cfg:PiperPickNPlaceEnvCfg",
        # if use PPO in rl_games
        #"rl_games_cfg_entry_point": "project_831.tasks.agents:rl_games_ppo_cfg.yaml",
    },
)
