#!/bin/bash


up_down_radio=(0 0.25 0.5 0.75 1)
single_sideBySide=(0 0.5 1)


for u_d_r in "${up_down_radio[@]}"; do
  for s_s_b in "${single_sideBySide[@]}"; do
    echo "Running up_down_radio: $u_d_r, single_sideBySide: $s_s_b"
    python3 main.py --silent --visualize --up_down_radio $u_d_r --single_sideBySide $s_s_b
  done
done

