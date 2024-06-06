```sh
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
conda install cudnn=8.2 -c anaconda -y
```


```sh
catkin config --cmake-args -DPYTHON_EXECUTABLE=$ANACONDA_PATH/envs/$ENV_NAME/bin/python3.9 -DPYTHON_INCLUDE_DIR=$ANACONDA_PATH/envs/$ENV_NAME/include/python3.9 -DPYTHON_LIBRARY=$ANACONDA_PATH/envs/$ENV_NAME/lib/libpython3.9.so

```