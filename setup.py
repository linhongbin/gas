from setuptools import setup, find_packages

setup(name='gym_ras', 
      version='1.0',
    install_requires=[
        'gym<=0.24', 
        'opencv-python<4.10',
        'ruamel.yaml<=0.17',
        'tqdm',
        'Pillow',
        'scikit-image',
        'pandas>=0.18',
        'pynput',
        'pybullet==3.0.9',
        'numpy<=1.23',
        'setuptools<=59.5.0',
        ], 
      packages=find_packages())
