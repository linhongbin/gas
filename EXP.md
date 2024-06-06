
calibrate dvrk ws
```sh
python ./gym_ras/run/calibrate_dvrk_ws.py
```

stream cam
```sh
python ./gym_ras/run/stream_cam.py --seg-dir ./data/dvrk_cal/color_seg_cal.yaml --seg-type color --vis-tag rgb mask
```


Keyboard
```sh
python ./gym_ras/run/env_play.py --env-tag gas_surrol dvrk_grasp_any
```


<!-- Baseline dVRK

gas
```sh
 python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS/2024_01_21-13_57_13@ras-gas_surrol@dreamerv2-gas@seed1/ --reload-envtag gas_surrol dvrk_grasp_any    --online-eval --visualize --vis-tag obs rgb mask
```


ppo
```sh
 python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/ppo/2024_01_26-23_55_16@ras-gas_surrol-raw-no_clutch@ppo@seed2/ --reload-envtag gas_surrol raw no_clutch dvrk_grasp_any    --online-eval --visualize --vis-tag obs rgb mask
``` -->
==========================================================


GAS: `python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS/2024_01_21-13_57_13@ras-gas_surrol@dreamerv2-gas@seed1 --reload-envtag gas_surrol dvrk_grasp_any --online-eval --visualize --vis-tag obs rgb mask --online-eps --save-prefix `

<!-- GAS-NoDE: `python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS-NoDE/2024_01_25-01_34_32@ras-gas_surrol-no_depth_estimation@dreamerv2-gas@seed1/ --reload-envtag gas_surrol dvrk_grasp_any no_depth_estimation --online-eval --visualize --vis-tag obs rgb mask --online-eps --save-prefix` -->

GAS-NoDR: `python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS-NoDR/2024_01_26-15_23_06@ras-gas_surrol-no_dr@dreamerv2-gas@seed1/ --reload-envtag gas_surrol dvrk_grasp_any no_dr --online-eval --visualize --vis-tag obs rgb mask --online-eps --save-prefix`

<!-- GAS-NoClutch: `python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS-NoClutch/2024_01_25-01_35_16@ras-gas_surrol-no_clutch@dreamerv2-gas@seed1 --reload-envtag gas_surrol dvrk_grasp_any no_clutch --online-eval --visualize --vis-tag obs rgb mask --online-eps --save-prefix`

GAS-Raw: `python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS-Raw/2024_01_25-12_26_44@ras-gas_surrol-raw@dreamerv2-gas@seed2/ --reload-envtag gas_surrol dvrk_grasp_any raw --online-eval --visualize --vis-tag obs rgb mask --online-eps --save-prefix` -->

dreamerv2: `python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/dreamerv2/2024_01_25-12_23_27@ras-gas_surrol-raw-no_clutch@dreamerv2-gas@seed1/ --reload-envtag gas_surrol dvrk_grasp_any raw no_clutch --online-eval --visualize --vis-tag obs rgb mask --online-eps --save-prefix`

PPO: `python ./gym_ras/run/rl_train.py --reload-dir  ./data/agent/PPO/2024_01_26-23_55_16@ras-gas_surrol-raw-no_clutch@ppo@seed2/ --reload-envtag gas_surrol dvrk_grasp_any raw no_clutch   --online-eval --visualize --vis-tag obs rgb mask --online-eps --save-prefix`


===========================

*Generality study*
`python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS/2024_01_21-13_57_13@ras-gas_surrol@dreamerv2-gas@seed1 --reload-envtag gas_surrol dvrk_grasp_any --online-eval --visualize --vis-tag obs rgb mask --online-eps 20 --save-prefix `

===========================

*disturbance study* 

sythetic image noise
`python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS/2024_01_21-13_57_13@ras-gas_surrol@dreamerv2-gas@seed1 --reload-envtag gas_surrol dvrk_grasp_any image_noise_on --online-eval --visualize --vis-tag obs rgb mask --online-eps 20 --save-prefix `


action noise
`python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/GAS/2024_01_21-13_57_13@ras-gas_surrol@dreamerv2-gas@seed1 --reload-envtag gas_surrol dvrk_grasp_any image_noise_on --online-eval --visualize --vis-tag obs rgb mask --online-eps 20 --save-prefix `