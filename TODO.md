# 1. Gym-ras development



## 1.1. surrol dev

- [ ] ~~Multi-thread surrol data generation~~

- [x] filter out realistic grasp, for example, a) needle slip into grippers (which meets the grasping codition) while gripper is closed b) gripper poke the ground (workspace z-axis minimum is too low)
- [x] create new task that grasp any shape of object 
- [x] add negative reward on object sliding when gripper pushes
- [x] add negative reward on gripper toggle times
- [x] add disturbance on stuff
- [x] increase lift height success threshold
- [x] negative reward if out of dsa zoom


## 1.2. Depth Estimation

- [ ] stereo depth estimation (yonghao) :large_blue_circle:
- ~~[ ] post processing depth image to eliminate noise (maybe):hourglass_flowing_sand:~~
- ~~[ ] challenge on tinny object with realsense435~~

## 1.3. image representation

- [x] general simple 2, with no depth image input, one camera
- [x] general simple 2, with no depth image input, two camera
- [ ] ~~goal representation for general simple2~~
- [x] dsa, global image, rectangle increase, dsa_3

## 1.4. Segmentation

### 1.4.1. using private dataset
- [ ] autonomate program to get mask from green background
- [ ] synthesize background image
- [ ] include stuffs in diverse task, (peg , gauze, etc.)
- [ ] ~~manual annote with eiseg~~
- [ ] maual annote with cvat 
### 1.4.2. using public dataset
- [ ] train with online dataset (junwei):large_blue_circle:



### 1.4.3. enhance
- [ ] use data augmentation in detectron2 (junwei) :large_blue_circle:

##  1.5. Domain randomization
### 1.5.1. image noise
- [X] random texture on ground, `surrol` 
- [x] circle cutout
- [x] rectange cutout
- [x] line cutout
- [x] gaussian blur
- [x] pepper and sault noise
 
### 1.5.2. cam pose noise
- [x] random reset camera pose
- [x] random disturbance on camera pose 

### 1.5.3. gripper action noise
- [x] random step action small transformation  


## 1.6. surgical stage recognition

- [ ] state estimation for multiple stages in a task (Li Bin) :large_blue_circle:

## 1.7. affordance

## 1.8. depth process

- [x] depth uncertain drift
- [x] depth uncertain range


## 1.9. action smooth

- [x] add ema filter to the cartestian desire pose

## 1.10. dVRK grasping task

- target task: 
    (a) gauze retrieve, 
    (b) cotton sponge retrieve, 
    (c) surgical debridegement (retrieve tissue), 
    (d) needle picking 
    (e) surgical-thread picking

## 1.11. Baseline

- [ ] pose-related signals :red_circle:
- [ ] ppo  image+vector :red_circle:
- [ ] ppo  pose-related signals :red_circle:


# 2. Experiment

## 2.1. surrol: 

### 2.1.1. dreamerv2 cnn 
```sh
--baseline-tag gym_ras_np --baseline dreamerv2
```
~~- [ ] standard training:~~
~~- [ ] realistic training: `--env-tag cam_pose_noise_medium depth_remap_noise_medium action_noise_low`~~
- [x] no oracle: `--baseline-tag no_oracle` [log](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EempYszUBZRKkDzF_K9zNsgBiSu1BQAZGuKcjud_sXTOtg?e=34pM7e)
- [x] medium domain adaptation: `--env-tag cam_pose_noise_medium depth_remap_noise_medium` [log](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EeQgKp95g9VLn4KcSuiu1_ABG4nhQe8ysL-4lyiEnq4Kpw?e=Crky0Q)
- [x] high domain adaptation (no action noise): `--env-tag cam_pose_noise_high depth_remap_noise_high action_noise_high` [log](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EXKXTQfj9gBPicQClg3hBXEBVqrHgCohJGswB7qRxHg50Q?e=fUIfe8)
-  ~~cam_rgbm, standard training: `--env-tag cam_rgbm`~~
- ~~cam_rgbm, realistic training: `--env-tag cam_rgbm cam_pose_noise_medium depth_remap_noise_medium ~~action_noise_low`~~
- ~~[ ] cam_rgbm two cam, realistic training:  `--env-tag cam_rgbm_dual cam_pose_noise_medium depth_remap_noise_medium action_noise_low`~~

- [x] rgbd, image_noise, realistic training, no_depth_link, mask_except_gripper:  `--env-tag image_noise cam_pose_noise_medium depth_remap_noise_medium action_noise_low no_depth_link mask_except_gripper` [log](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EWFeSVwg2elGr0WY8I7dXQUBgQoOt7PAAVF6TfTYvwNPXg?e=dCaaoT)

- ~~[ ] rgbd, image_noise_low, realistic training, no_depth_link, :  `--env-tag image_noise_low cam_pose_noise_medium depth_remap_noise_medium action_noise_low no_depth_link`:hourglass_flowing_sand:~~


- ~~[ ] rgbd, image_noise_real, realistic training, no_depth_link, depth_process:  `--env-tag no_depth_link cam_pose_noise_medium depth_remap_noise_medium image_noise_real depth_process`--:hourglass_flowing_sand:~~

- [x] rgbd, dvrk_cam_setting, no segment noise, :  `--env-tag dvrk_cam_setting` [log1](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EbQe7L3zLwBPnK09sKEH1G4B-1aX0TnTKQnYWLEG9Aw-Yg?e=jVu5ro) (before filter out relaistic grasp), [2023_11_29-20_07_35@ras-dvrk_cam_setting@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/ER0tGgBLcqpKo_2BPidfCNsBMH6k1jctX_u9fuipKsfgNQ?e=F22LeO)(after applying realistic grasp)

- [x] rgbd, dvrk_cam_setting, no segment noise, dsa_zoom_high:  `--env-tag dvrk_cam_setting dsa_zoom_high` [2023_11_29-20_09_58@ras-dvrk_cam_setting-dsa_zoom_high@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EchF3chSjSxAqgmpj9lbC5oBp7SwXXEHOVD9o6UOPU28kA?e=A6XQSQ)

- [x] rgbd, dvrk_cam_setting, no segment noise, action_smooth :  `--env-tag dvrk_cam_setting action_smooth` [2023_11_30-11_12_20@ras-dvrk_cam_setting-action_smooth@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EdyQs4sjtexCt3--h_ifXkgBS5wRdSaQelOertG6fVmyNw?e=DO552Q)

-  ~~rgbd, dvrk_cam_setting, no segment noise, dsa_zoom_high, dvrk large init random pose :  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high`~~ 

- ~~[ ] rgbd, dvrk_cam_setting, no segment noise, dsa_zoom_high, dvrk large init random pose action_smooth:~~  ~~`--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high action_smooth`~~ 

- ~~[ ]`--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high reward2 large_action_scale`~~


- ~~[ ] `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high reward2 large_action_scale image_noise_real`~~

- [x]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high large_action_scale` [2023_12_14-18_47_27@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-large_action_scale@dreamerv2-gym_ras_np](
https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EYnJVZa6SkdNqMbBa4SP8oMB91GhbuX8UPk_1g9dPaCG2Q?e=kRqFiG)
- [x]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real` [2023_12_15-22_50_42@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EVEBnO7-qOxAoniBVaGYCLgB89CLAKA5rRSr4fQoFVRZlQ?e=06WhX5)

- [x]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real disturbance` **hard to converge**
- [ ]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real_high` 

- [x]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real reward2 dsa_3` 
[2023_12_24-14_33_34@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-reward2-dsa_3@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/ET_dapaxoNZGi4rokuaFRmgBcg6E5oICiswsmsDki659rA?e=k9a5Pi) slight slow converge

- [x]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real dsa_3` [2023_12_23-15_56_56@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-dsa_3@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/Ee4eVONjfrRJnIOtUZzVRgwBRSlbATNvtNSD4JCFLvfdKQ?e=Uh8GKm)

- [x]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real cam_dynamic dsa_3` [2023_12_24-14_35_13@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-cam_dynamic-dsa_3@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EUh9Qpbe4yFKhIhsgblLjCcBhGKN1qTZAaPlqdQ-ctn0NQ?e=JlXIEF)

- [x]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real cam_dynamic dsa_3 reward3` [2023_12_29-20_28_55@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-cam_dynamic-dsa_3-reward3@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EdT8ol7UhCNBmmB7tk7TyWUBXg5yb-acIXFfeRp_gzeEfw?e=H2iQjW) |[2024_01_08-23_14_42@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-cam_dynamic-dsa_3@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EdDm5LHcOi1JgfBcEwvuc_oBpiLYmo2qwrRgQZZKeDl6tA?e=3w7jNQ) | [2024_01_08-14_41_31@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-cam_dynamic-dsa_3@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EXqMaMR9ZmpFgfK-V9BZ2psBVneYZy2b1azfbnT80G0ymA?e=ebQ1im) 
- [ ]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting_grasp_any dsa_zoom_high grasp_any image_noise_real cam_dynamic dsa_3` :red_circle:

- [ ]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real cam_dynamic dsa_3 vec_obs2` [2024_01_04-15_14_36@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-cam_dynamic-dsa_3-vec_obs2@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EcfIDLrk635AsclfUJN8E8sBTEqJMDEl_KHD0wvBjNMs7A?e=ebckhD)


- [ ]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real dsa_3 dsa_anomaly reward4`  [2024_01_09-11_24_08@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-dsa_3-dsa_anomaly-reward4@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EdZbnI1bpzJOlyhf-n__3CIBhHrLtfyz3xfDqtZvSo64mA?e=9PtreG) |[2024_01_09-11_22_34@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-dsa_3-dsa_anomaly-reward4@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/Edk58o0EiLZCo3s6WcL1KNABL1Z8XmOEfvCxKNx7VYE2_A?e=dr8XYX)| [2024_01_09-11_19_44@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-dsa_3-dsa_anomaly-reward4@dreamerv2-gym_ras_np](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/Ec8cHrDAGGBDsReJBJr9dB4Bc7cAfZ5ZMJ7FLdvfn56Low?e=0HDHCF) 

### 2.1.2. dreamerv2 cnn no_oracle

```sh
--baseline-tag gym_ras_np no_oracle --baseline dreamerv2
```

- [ ]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real cam_dynamic dsa_3` :red_circle:

### 2.1.3. dreamerv2 cnn+mlp 

- [x]  `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real cam_dynamic dsa_3 vec_obs2 --baseline dreamerv2 --baseline-tag gym_ras_np mlp` [2023_12_28-15_42_44@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-cam_dynamic-dsa_3-vec_obs2@dreamerv2-gym_ras_np-mlp](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EQXcbwixu8RJnpt3fpk-XEcBKHMnJn5UFJNoP3BNlo4_mw?e=gCahMm) converge slightly lower than only cnn

### 2.1.4. PPO cnn+mlp


- [ ] `--env-tag dvrk_cam_setting dvrk_init_pose_setting dsa_zoom_high grasp_any image_noise_real cam_dynamic dsa_3 vec_obs2 --baseline ppo --baseline-tag gym_ras_np mlp`  :red_circle:
### 2.1.5. gauze retreive
```sh
baseline-tag gym_ras_np baseline dreamerv2
```
- [x] realistic training: `--env-tag cam_pose_noise_medium depth_remap_noise_medium action_noise_low` [log](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/Eeg9OFWYjP5HjyM1SCgVOY4BCz-yeHpAIvKLdbP3rStiRw?e=jPpfpV)





## 2.2. dVRK
### 2.2.1. needle pick
#### 2.2.1.1. dreamerv2

- [x] color seg, dvrk_cam_setting (succes rate 60%)
  
    The first success demo on needle picking on real robot!!!!
    ```sh
    python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/2023_11_24-01_03_32@ras-dvrk_cam_setting@dreamerv2-gym_ras_np/  --reload-envtag dvrk_real_dsa color_seg large_action_scale dvrk_cam_setting  --online-eval --visualize --vis-tag obs rgb
    ```

```sh
python ./gym_ras/run/rl_train.py --reload-dir ./data/agent/2023_12_29-20_28_55@ras-dvrk_cam_setting-dvrk_init_pose_setting-dsa_zoom_high-grasp_any-image_noise_real-cam_dynamic-dsa_3-reward3@dreamerv2-gym_ras_np  --reload-envtag dvrk_real_dsa color_seg  dvrk_cam_setting dsa_zoom_high grasp_any dsa_3  --online-eval --visualize --vis-tag obs rgb
```

# 3. RSS2024

- domain: general surgery grasping
- Name: GAS: Grasp Anything for Surgery

baselines

- DDPG, PPO, SAC on pose-tracking signals
- PPO, dreamerv2 on image
- our
  
Experiment


GAS



- `--baseline-tag gym_ras_np high_oracle --baseline dreamerv2  --env-tag surrol_grasp_any dsa_anomaly reward4 dvrk_init_pose_setting_grasp_any disturbance2 action_noise_low2`[2024_01_11-15_13_36@ras-surrol_grasp_any-dsa_anomaly-reward4-dvrk_init_pose_setting_grasp_any-disturbance2@dreamerv2-gym_ras_np-high_oracle](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EV9wTCkIPr1Ai3qMs7VTIPUBjIABd5f5oGq9wOCKrT7Maw?e=oDwQaD)


- `--baseline-tag gym_ras_np high_oracle --baseline dreamerv2  --env-tag surrol_grasp_any dsa_anomaly reward5 dvrk_init_pose_setting_grasp_any disturbance2 action_noise_low2`[2024_01_12-17_58_57@ras-surrol_grasp_any-dsa_anomaly-reward5-dvrk_init_pose_setting_grasp_any-disturbance2-action_noise_low2@dreamerv2-gym_ras_np-high_oracle](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/ES880b7lxopBpfa4VnwkjocBlkb8SFdFvkFKj7FXnyLbeg?e=Fp65is)


- `--baseline-tag gym_ras_np high_oracle --baseline dreamerv2  --env-tag surrol_grasp_any dsa_anomaly reward5 dvrk_init_pose_setting_grasp_any disturbance2 action_noise_low2 vector2image_square`[2024_01_14-20_31_54@ras-surrol_grasp_any-dsa_anomaly-reward5-dvrk_init_pose_setting_grasp_any-disturbance2-action_noise_low2-vector2image_square@dreamerv2-gym_ras_np-high_oracle](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/ERPH0ARC5rxFskDtWf_-cz0B-y8JgDG_IXtD2EBsCLj42A?e=3j4u2K) 90% succcess rate

- `--baseline-tag gym_ras_np high_oracle --baseline dreamerv2  --env-tag surrol_grasp_any dsa_anomaly reward6 dvrk_init_pose_setting_grasp_any disturbance2 action_noise_low2 vector2image_square`[2024_01_14-00_17_13@ras-surrol_grasp_any-dsa_anomaly-reward6-dvrk_init_pose_setting_grasp_any-disturbance2-action_noise_low2@dreamerv2-gym_ras_np-high_oracle](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EaTr0Qeja8lGgDODOKwOU2IBAD8wRbnJQtNGE5J_BqdHyw?e=EbPG3d) 70% success rate

- `--baseline-tag gym_ras_np high_oracle --baseline dreamerv2  --env-tag surrol_grasp_any dvrk_init_pose_setting_grasp_any disturbance2 action_noise_low2`[2024_01_14-00_22_40@ras-surrol_grasp_any-dvrk_init_pose_setting_grasp_any-disturbance2-action_noise_low2@dreamerv2-gym_ras_np-high_oracle](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/ESZ1rsxzS25HhA2-G8S2ZTcBQgTRw98qeMzLLVhWUTgoNQ?e=QhoS2m) 60% success rate

- `--baseline-tag gym_ras_np high_oracle --baseline dreamerv2  --env-tag surrol_grasp_any dvrk_init_pose_setting_grasp_any disturbance2 action_noise_low2 insertion_anamaly`[2024_01_15-18_12_50@ras-surrol_grasp_any-dsa_anomaly-reward5-dvrk_init_pose_setting_grasp_any-disturbance2-action_noise_low2-insertion_anamaly@dreamerv2-gym_ras_np-high_oracle](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EafR-BqtlQhAt69R29KdqzIBDSqAQlr6kAFJ8uLcDr2KZg?e=H0L9rs) 80% success rate


## Paper Experiement

- GAS , Surrol
  `python ./gym_ras/run/rl_train.py --env-tag gas_surrol --baseline dreamerv2 --baseline-tag gas`

- GAS-Raw ,Surrol
  `python ./gym_ras/run/rl_train.py --env-tag gas_surrol raw --baseline dreamerv2 --baseline-tag gas`

- GAS-NoDE ,Surrol
  `python ./gym_ras/run/rl_train.py --env-tag gas_surrol no_depth_estimation --baseline dreamerv2 --baseline-tag gas`

- GAS-NoClutch ,Surrol
  `python ./gym_ras/run/rl_train.py --env-tag gas_surrol no_clutch --baseline dreamerv2 --baseline-tag gas`

- GAS-NoDR ,Surrol
  `python ./gym_ras/run/rl_train.py --env-tag gas_surrol no_dm --baseline dreamerv2 --baseline-tag gas`


- DreamerV2 , Surrol
  `python ./gym_ras/run/rl_train.py --env-tag gas_surrol raw no_clutch --baseline dreamerv2 --baseline-tag gas`

- PPO , Surrol
  `python ./gym_ras/run/rl_train.py --env-tag gas_surrol raw no_clutch --baseline ppo`