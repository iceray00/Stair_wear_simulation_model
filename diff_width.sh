#!/bin/bash


width_diff=(30 40 50 60)


for wid in "${width_diff[@]}"; do
  echo -e "Running width: $wid \n  up_down_radio: 0.75, single_sideBySide: 0.4"
  python3 main.py --silent --visualize --up_down_radio 0.75 --single_sideBySide 0.4 --width $wid
done

