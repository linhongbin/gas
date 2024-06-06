from setuptools import setup, find_packages

setup(name='gym_ras', 
      version='1.0',
    install_requires=[
        'gym<=0.24', 
        'opencv-python',
        'ruamel.yaml<=0.17',
        'tqdm',
        'Pillow',
        'scikit-image',
        'pandas>=0.18',
        'pynput',
        ], 
      packages=find_packages())
