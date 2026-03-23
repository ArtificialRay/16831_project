python3 scripts/sac/train.py \
    --task PiperPickNPlace \
    --agent sac_cfg_entry_point \
    --save-frequency 2000 \
    --eval-frequency 1000 \
    --log_dir logs/sac