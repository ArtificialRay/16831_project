# Installation

## Create New Project
Inside IsaacLab root direction, run: 
```bash
./isaaclab.sh --new
```
Choose agent type: single-agent; RL module: rl-game

## Install project
Inside project831, run: 
```bash
python -m pip install -e ./source/project_831
```

# Launching in IsaacLab

## Command for Launching Random Robot Agent
```bash
python scripts/random_agent.py --task PiperPickNPlace
```

There is one hard-code path at ./source/project_831/project_831/tasks/piper_env_cfg.py, change to your-project-path/assets/robots/