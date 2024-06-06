# World Models for General Surgical Grasping

Implementation of the [GAS][website] agent.
Simulation and real robot (dVRK) are supported.

## Install

- Create conda virtual environment
```sh
source ./config.sh
source $ANACONDA_PATH/bin/activate 
conda create -n $ENV_NAME python=3.7 -y
```

- Install our gym package 
```sh
pushd ext/SurRoL/
python -m pip install -e . # install surrol
popd
python -m pip install -e . # install gym_ras
```



