pushd ext/SurRoL/
python -m pip install -e . # install surrol
popd
python -m pip install -e . # install gym_ras
conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y # install cuda-toolkit for gpu support
conda install ffmpeg -y
pushd ext/dreamerv2/
python -m pip install -e .
popd