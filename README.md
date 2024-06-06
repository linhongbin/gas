# 1. World Models for General Surgical Grasping

Implementation of the [GAS][website] agent.
Simulation and real robot (dVRK) are supported.

# 2. Install

## 2.1. Download
```sh
git clone https://github.com/linhongbin/gas.git
cd gas
git submodule update --init --recursive
```

## 2.2. Environment
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

## 2.3. Training 

```sh
conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y # install cuda-toolkit for gpu support
conda install ffmpeg -y
#python -m pip install tensorflow==2.9.0 tensorflow_probability==0.17.0 protobuf==3.20.1
pushd ext/dreamerv2/
python -m pip install -e .
popd
```

# 3. Run

## 3.1. Train

Train `GAS`
```sh
source init.sh
python ./run/rl_train.py --env-tag gas_surrol --baseline dreamerv2 --baseline-tag gas
```

Train other baselines 
(baseline names are commented at the end of command lines)
```sh
python ./run/rl_train.py --env-tag gas_surrol raw --baseline dreamerv2 --baseline-tag gas # GAS-Raw
python ./run/rl_train.py --env-tag gas_surrol no_depth_estimation --baseline dreamerv2 --baseline-tag gas # GAS-NoDE
python ./run/rl_train.py --env-tag gas_surrol no_clutch --baseline dreamerv2 --baseline-tag gas # GAS-NoClutch
python ./run/rl_train.py --env-tag gas_surrol no_dm --baseline dreamerv2 --baseline-tag gas # GAS-NoDR
python ./run/rl_train.py --env-tag gas_surrol raw no_clutch --baseline dreamerv2 --baseline-tag gas # DreamerV2
python ./run/rl_train.py --env-tag gas_surrol raw no_clutch --baseline ppo # PPO
```

Mornitor training with tensorboard, open a new terminal
```sh
source init.sh
tensorboard --logdir ./log/
```


Pretrained models of baselines can be downloaded here:
|             | Model | Sim-SR | Real-SR |
| :------:   | :------: | :------: | :------: | 
|     GAS    |     [Download]([link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155097177_link_cuhk_edu_hk/EXRee1wtjxNBtxEKjQDay7kB15cl58-LBgRTRlqjJp6Phg?e=v2rbF2&download=1))     |           |           |
|  GAS-NoDR   |     [Download](link)    |          |            |
|  GAS-Raw   |     [Download](link)    |          |            |
|  GAS-NoDE   |     [Download](link)    |          |            |
|  GAS-NoClutch   |     [Download](link)    |          |            |
|  DreamerV2   |     [Download](link)    |          |            |
|  PPO   |     [Download](link)    |          |            |


## 3.2. Evaluation

### 3.2.1. Simulation

### 3.2.2. Real Robot


# 4. Acknowledgement