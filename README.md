# 1. World Models for General Surgical Grasping

Implementation of the "Grasp Anything for Surgery" (GAS).
[[project website](http://hongbinlin.com/gas/)] [[paper](https://arxiv.org/pdf/2405.17940)]

# 2. Install

## 2.1. Download
```sh
git clone -b main --single-branch https://github.com/linhongbin/gas.git
cd gas
git submodule update --init --recursive
```

## 2.2. Conda Install

- We use virtual environment in [Anancoda](https://www.anaconda.com/download) to install our codes. Make sure you install anaconda.

- Edit environment variables, go to [config.sh](./config.sh) and edit your environment variables.

- Create conda virtual environment
    ```sh
    source ./config.sh
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.9 -y
    ```

- Install our gym package 
    ```sh
    source init.sh
    conda install libffi==3.3 -y
    pushd ext/SurRoL/
    python -m pip install -e . # install surrol
    popd
    python -m pip install -e . # install gym_ras
    ``` 
- Install the RL backbone: DreamerV2
    ```sh
    conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y # install cuda-toolkit for gpu support
    conda install ffmpeg -y
    pushd ext/dreamerv2/
    python -m pip install -e .
    popd
    ```
    > Our GPU Dependency in Anaconda: cudatoolkit=11.3, cudnn=8.2,tensorflow=2.9.0 tensorflow_probability=0.17.0. Other versions should work, while user need to check the compatability of cuda and tensorflow.  

# 3. Run

## 3.1. Train

- Train our visuomotor controller `GAS`
    ```sh
    source init.sh
    python ./run/rl_train.py --env-tag gas_surrol --baseline dreamerv2 --baseline-tag gas
    ```

- Train other baselines (names are commented at the end of command lines)
    ```sh
    python ./run/rl_train.py --env-tag gas_surrol raw --baseline dreamerv2 --baseline-tag gas # GAS-Raw
    python ./run/rl_train.py --env-tag gas_surrol no_depth_estimation --baseline dreamerv2 --baseline-tag gas # GAS-NoDE
    python ./run/rl_train.py --env-tag gas_surrol no_clutch --baseline dreamerv2 --baseline-tag gas # GAS-NoClutch
    python ./run/rl_train.py --env-tag gas_surrol no_dm --baseline dreamerv2 --baseline-tag gas # GAS-NoDR
    python ./run/rl_train.py --env-tag gas_surrol raw no_clutch --baseline dreamerv2 --baseline-tag gas # DreamerV2
    python ./run/rl_train.py --env-tag gas_surrol raw no_clutch --baseline ppo # PPO
    ```

- Mornitor training with tensorboard, open a new terminal
    ```sh
    source init.sh
    tensorboard --logdir ./log/
    ```

- Pretrained Models
  
    Pretrained models that used in our paper can be downloaded here. Sucess rate (SR) in simulation and real robots for our performance study in the paper are also listed:
    |             | Model | Sim-SR | Real-SR |
    | :------:   | :------: | :------: | :------: | 
    |     GAS (ours)   |     [Download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EXRee1wtjxNBtxEKjQDay7kB15cl58-LBgRTRlqjJp6Phg?e=v2rbF2&download=1)     |        87%   |      75%     |
    |  GAS-NoDR   |     [Download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EfRwARbAi1VIhrPn1ESWUnkB_9F5wU6SyDJ8xs-VEhSyLQ?e=O2cfF5&download=1)    |   84%       |     58%       |
    |  GAS-NoClutch   |     [Download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EZn1Ei7XTPFHj1J1muXZE5QBHMALIeUFs1A7BVkdUqWN7g?e=RAhbFx&download=1)    |     36%   |       0%     |
    |  GAS-Raw   |     [Download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EX_g_ph0I65JhUIje_wHLs4BW7G8ybm9K0rghlGSMzqjDQ?e=ZbKCZ0&download=1)    |      0%    |       0%     |
    |  GAS-NoDE   |     [Download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EXpRpGEgxYNPpUd2CDTHLwMBIoIXyEkrjnHYJGa8yQKypA?e=Eoqatc&download=1)    |     0%     |     0%       |
    |  DreamerV2   |     [Download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EbhCutr_fr1KiLrkq6BKwnAB4th_WnR3uq8qLR7xB2ycWQ?e=f7S2BM&download=1)    |      0%    |        1%      |
    |  PPO   |     [Download](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EcvH8m0OpMFBmiSONJaxsdgB5Qv3jGd8YOGo0s41b6EwEQ?e=OFSv8m&download=1)    |      0%    |         2%   |


## 3.2. Evaluation

### 3.2.1. Simulation

- Run evaluation of GAS (ours) in simulation:
    ```sh
    source init.sh
    python ./run/rl_train.py --reload-dir ./log/2024_01_21-13_57_13@ras-gas_surrol@dreamerv2-gas@seed1/ --reload-envtag gas_surrol --online-eval --visualize --vis-tag obs rgb mask --online-eps 20 --save-prefix GAS
    ```
### 3.2.2. Real Robot


# 4. Acknowledgement