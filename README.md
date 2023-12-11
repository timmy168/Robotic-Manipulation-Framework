# Robotic-Manipulation-Framwork
In this wok, implemented both forward kinematics (FK) and inverse kinematics (IK) functions to determine joint angles and velocities for controlling a 6-degree-offreedom robot arm. The PyBullet simulation environment was employed to validate the functionality of these kinematic functions. Specifically, manipulated a UR5 robot arm, utilizing the simulation to perform a block insertion task as part of the verification process.

# Requirement
(1) Ubuntu 20.04
(2) Python 3.7
(3) Ravens

# Installation

Follow the installation instruction in https://github.com/google-research/ravens

1. Create and activate Conda environment, then install GCC and Python packages.

```shell
git pull
cd hw4
cd ravens
conda create --name pdm-hw4 python=3.7 -y
conda activate pdm-hw4
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
pip install -r requirements.txt
```
2. Install GPU acceleration with NVIDIA CUDA 10.1 and cuDNN 7.6.5 for Tensorflow.
```bash
conda install cudatoolkit==10.1.243 -y
conda install cudnn==7.6.5 -y
```
