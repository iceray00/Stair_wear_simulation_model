
# Decoding Staircase Histories: Bridging Stair Wear Simulation Models and CNN Predictions


## Dependencies

```bash
pip3 install -r requirements.txt
```


## Quick Start

* As default, you can run:
```bash
python3 main.py
```

And the other more detailed strategies, the detailed content is presented as follows:

```bash
usage: main.py [-h] [-l LENGTH] [-w WIDTH] [--height HEIGHT] [--mu_0 MU_0] [--mu_min MU_MIN] [--theta THETA] [-v V] [--materials MATERIALS] [-k K] [-F F] [-H H]
               [--silent] [--visualize] [--up_down_radio UP_DOWN_RADIO] [--single_sideBySide_radio SINGLE_SIDEBYSIDE_RADIO]
               [--stair_use_per_day STAIR_USE_PER_DAY] [--use_year USE_YEAR] [--aggregate AGGREGATE]

options:
  -h, --help            show this help message and exit
  -l LENGTH, --length LENGTH
                        Length of the stairs, in cm. Default = 150
  -w WIDTH, --width WIDTH
                        Width of the stairs, in cm. Default = 30
  --height HEIGHT       Height of the stairs, in cm. Default = 15
  --mu_0 MU_0           Surface friction coefficient. Default = 0.5
  --mu_min MU_MIN       Minimum friction coefficient at depth. Default = 0.4
  --theta THETA         Angle of the stairs in degrees. Default = 26.566. Meaning tan(26.566) = 1/2
  -v V, --v V           Speed of the people who walking the stairs. Default = 0.8
  --materials MATERIALS
                        Material Name of the stairs
  -k K, --k K           Wear coefficient. Default = 10**(-4)
  -F F, --F F           Normal load of the stairs, in N. Default = 600
  -H H, --H H           Hardness of the stairs, in MPa. Default = 200
  --silent              Silent mode
  --visualize           Visualize the stairs
  --up_down_radio UP_DOWN_RADIO
                        Up and down ratio of the stairs. `0.2` meaning 20 percent up and 80 percent down. 0 <= up_down_radio <= 1. Default = 0.5
  --single_sideBySide_radio SINGLE_SIDEBYSIDE_RADIO
                        Probability of walking side by side (0 = all walking alone). 0 <= single_sideBySide_radio <= 1. Default = 0.5
  --stair_use_per_day STAIR_USE_PER_DAY
                        Number of stairs used per day. Default = 19000
  --use_year USE_YEAR   Number of years. Default = 400
  --aggregate AGGREGATE
                        Number of stairs aggregated. Default = 1000000
```









