# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

import gymnasium as gym

from .piper_env import PiperSwingEnv
from .piper_env_cfg import PiperSwingEnvCfg

gym.register(
    id="project831-PiperSwing-v0",
    entry_point="project_831.tasks.piper_env:PiperSwingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "project_831.tasks.piper_env_cfg:PiperSwingEnvCfg",
    },
)

