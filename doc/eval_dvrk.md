# Install

- ROS: Install ROS melodic for Ubuntu 18.04 or noetic for Ubuntu 20.04
- Modify [config_dvrk.sh](../config_dvrk.sh)
- Create a new conda environemnt
  ```sh
  source config_dvrk.sh
  conda create -n $ENV_NAME python=3.9 -y
  conda activate gas_dvrk
  ```
- Install Conda Dependency
    ```
    conda install -c conda-forge ros-rospy wstool ros-sensor-msgs ros-geometry-msgs ros-diagnostic-msgs empy rospkg python-orocos-kdl -y 
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
    conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y 
    conda install ffmpeg libffi==3.3 -y
    ```
- dVRK Dependency
    - for melodic:
    ```sh
    sudo apt install libxml2-dev libraw1394-dev libncurses5-dev qtcreator swig sox espeak cmake-curses-gui cmake-qt-gui git subversion gfortran libcppunit-dev libqt5xmlpatterns5-dev  libbluetooth-dev python-wstool python-vcstool python-catkin-tools
    #conda install -c conda-forge ros-rospy wstool ros-sensor-msgs ros-geometry-msgs ros-diagnostic-msgs -y # additional install for melodic since it uses python2.7, need to reinstall all ros dependency
    ```
    - for noetic:
    ```sh
    sudo apt install libxml2-dev libraw1394-dev libncurses5-dev qtcreator swig sox espeak cmake-curses-gui cmake-qt-gui git subversion gfortran libcppunit-dev libqt5xmlpatterns5-dev libbluetooth-dev python3-pyudev python3-wstool python3-vcstool python3-catkin-tools python3-osrf-pycommon
    ```
- Install ROS dependency in Conda environment
    ```sh
    #conda install -c conda-forge empy rospkg python-orocos-kdl -y # install pre-compiled ros packages
    mkdir -p ext/ros_ws/src 
    pushd ext/ros_ws/src 
    git clone https://github.com/ros/geometry -b $ROS_DISTRO-devel 
    git clone https://github.com/ros/geometry2 -b $ROS_DISTRO-devel
    cd ..
    source /opt/ros/$ROS_DISTRO/setup.bash
    catkin config --cmake-args -DPYTHON_EXECUTABLE=$ANACONDA_PATH/envs/$ENV_NAME/bin/python3.9 -DPYTHON_INCLUDE_DIR=$ANACONDA_PATH/envs/$ENV_NAME/include/python3.9 -DPYTHON_LIBRARY=$ANACONDA_PATH/envs/$ENV_NAME/lib/libpython3.9.so
    catkin build # compile ros packages
    popd
    ```
- Install dVRK
    ```sh
    python -m pip install defusedxml numpy==1.23
    mkdir ./ext/dvrk_2_1/
    pushd ./ext/dvrk_2_1/
    catkin init
    catkin config --cmake-args -DPYTHON_EXECUTABLE=$ANACONDA_PATH/envs/$ENV_NAME/bin/python3.9 -DPYTHON_INCLUDE_DIR=$ANACONDA_PATH/envs/$ENV_NAME/include/python3.9 -DPYTHON_LIBRARY=$ANACONDA_PATH/envs/$ENV_NAME/lib/libpython3.9.so
    cd src
    vcs import --recursive --workers 1 --input https://raw.githubusercontent.com/jhu-saw/vcs/main/ros1-dvrk-2.1.0.vcsdevel
    catkin build --summary
    popd
    ```
- Install RealsenseD435 Driver
    ```sh
    dpkg -l | grep "realsense" | cut -d " " -f 3 | xargs sudo dpkg --purge # remove all realsense
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
    sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
    sudo apt-get update
    sudo apt-get install librealsense2-dkms librealsense2-utils -y
    sudo apt-get install ros-$ROS_DISTRO-realsense2-camera ros-$ROS_DISTRO-realsense2-description -y
    ```
- Install DualShock4 Joystick Driver
    ```sh
    source ./init_dvrk.sh
    pushd ./ext/
    git clone https://github.com/naoki-mizuno/ds4drv --branch devel
    cd ./ds4drv
    python -m pip install -e .
    sudo cp udev/50-ds4drv.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    popd
    pushd ./ext/ros_ws/src
    git clone https://github.com/naoki-mizuno/ds4_driver.git -b noetic-devel # Do not need to modify for melodic user, use noetic branch to support python3
    catkin build
    popd
    ```
- Install GAS package similar to [README.md](../README.md)
    ```sh
    #conda install libffi==3.3 -y
    pushd ext/SurRoL/
    python -m pip install -e . # install surrol
    popd
    python -m pip install -e . # install gym_ras
    #conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y # install cuda-toolkit for gpu support
    #conda install ffmpeg -y
    pushd ext/dreamerv2/
    python -m pip install -e .
    popd
    ```
- Install TrackAny
    ```sh
    #conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
    pushd ./ext/Track-Anything/
    python -m pip install -r requirements.txt
    popd
    ```

# Run

### 2.1. Launch roscore

- Open a terminal run
    ```sh
    source init_dvrk.sh
    roscore
    ```
### 2.2. launch dvrk console

- Simulation:
    ```sh
    source init.sh 
    roslaunch dvrk_robot dvrk_arm_rviz.launch arm:=PSM1 # simulation
    ```

- Real robot
    ```sh
    source init.sh 
    qlacloserelays
    roslaunch dvrk_robot dvrk_arm_rviz.launch arm:=PSM1 config:=${PWD}/ext/dvrk_2_1/src/cisst-saw/sawIntuitiveResearchKit/share/cuhk-daVinci-2-0/console-PSM1.json # real robot
    ```
### 2.3. launch RS435

- Simulation:

    Download [rosbag](https://drive.google.com/file/d/1VO9obXyEhYG61J_bmfNGst_2_5vo2al7/view?usp=sharing) and place it to `./data/dvrk-sim` 

    Open a terminal, and run
    ```sh
    source init.sh
    rosbag play -l ./data/dvrk_sim/rs435-sim.bag
    ```
- Real hardware:
    ```sh
    roslaunch realsense2_camera rs_camera.launch json_file_path:=./config/rs_high_density.json align_depth:=true filters:=spatial,temporal,decimation,disparity
    ```

### 2.4. launch Dualshock4

- real hardware
    ```sh
    source init.sh
    roslaunch ds4_driver ds4_driver.launch
    ```

## Calibrate dvrk workspace

```sh
source init.sh
python ./gym_ras/run/calibrate_dvrk_ws.py
```

## 

```sh
source init.sh
```