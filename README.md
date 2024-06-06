# 1. World Models for General Surgical Grasping

Implementation of the [GAS][website] agent.
Simulation and real robot (dVRK) are supported.

# 2. Install

## 2.1. Environment 
- Create conda virtual environment
```sh
source ./config.sh
source $ANACONDA_PATH/bin/activate 
conda create -n $ENV_NAME python=3.9 -y
```

- Install our gym package 
```sh
conda install libffi==3.3 -y
pushd ext/SurRoL/
python -m pip install -e . # install surrol
popd
python -m pip install -e . # install gym_ras
```

## 2.2. Training 

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
python ./run/rl_train.py --env-tag gas_surrol --baseline dreamerv2 --baseline-tag gas
```

## 3.2. Evaluation

### 3.2.1. Simulation

### 3.2.2. Real Robot


# 4. Acknowledgement