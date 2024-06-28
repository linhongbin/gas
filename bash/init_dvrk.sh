source ./config/config_dvrk.sh
source $ANACONDA_PATH/bin/activate
conda activate $ENV_NAME
alias python=$ANACONDA_PATH/envs/$ENV_NAME/bin/python3.9
export LD_LIBRARY_PATH=$ANACONDA_PATH/envs/$ENV_NAME/lib/:/usr/local/lib/
source ./ext/ros_ws/devel/setup.bash
source ./ext/dvrk_2_1/devel/setup.bash