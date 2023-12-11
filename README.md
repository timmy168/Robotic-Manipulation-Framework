# Robotic-Manipulation-Framwork
In this wok, implemented both forward kinematics (FK) and inverse kinematics (IK) functions to determine joint angles and velocities for controlling a 6 DOF robot arm. The PyBullet simulation environment was employed to validate the functionality of these kinematic functions. Specifically, manipulated a UR5 robot arm, utilizing the simulation to perform a block insertion task as part of the verification process.

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
conda create --name ravens python=3.7 -y
conda activate ravens
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
pip install -r requirements.txt
```
2. Install GPU acceleration with NVIDIA CUDA 10.1 and cuDNN 7.6.5 for Tensorflow.
```bash
conda install cudatoolkit==10.1.243 -y
conda install cudnn==7.6.5 -y
```
# Foward Kinematic Function
- Execution example
```bash
python fk.py
pybullet build time: Sep 22 2020 00:55:20
current_dir=/home/timmy/pdm-f23/hw4/pybullet_robot_envs/envs/panda_envs
============================ Task 1 : Forward Kinematic ============================

- Testcase file : fk_test_case_ta1.json
- Your Score Of Forward Kinematic : 3.333 / 3.333, Error Count :    0 /  300
- Your Score Of Jacobian Matrix   : 3.333 / 3.333, Error Count :    0 /  300

- Testcase file : fk_test_case_ta2.json
- Your Score Of Forward Kinematic : 3.333 / 3.333, Error Count :    0 /  100
- Your Score Of Jacobian Matrix   : 3.333 / 3.333, Error Count :    0 /  100

====================================================================================
- Your Total Score : 20.000 / 20.000
====================================================================================
```

# Inverse Kinematic Function
- Execution example
```bash
python ik.py

============================ Task 2 : Inverse Kinematic ============================

- Testcase file : ik_test_case_ta1.json
- Mean Error : 0.001048
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 20.000 / 20.000

- Testcase file : ik_test_case_ta2.json
- Mean Error : 0.001482
- Error Count :   0 / 100
- Your Score Of Inverse Kinematic : 20.000 / 20.000

====================================================================================
- Your Total Score : 40.000 / 40.000
====================================================================================
```

# Transport network manipulation pipeline

Test the ik implementation in the Transporter Networks 's frame work by inferencing "Block Insertion Task" on a mini set of 10 testing data

**Step 1.** Download dataset at https://drive.google.com/file/d/1JrCyrvpi3XeuapfecHs1aVtcfRJWWRTV/view?usp=drive_link and put the whole `block-insertion-easy-test/` folder under `hw4/ravens/`

**Step 2.** Download checkpoint at https://drive.google.com/file/d/1gEMNGTSXjMyvegp72ivbjCle0ntvcEVB/view?usp=drive_link and put the whole `checkpoints/` folder under `hw4/ravens/`

**Step 3.** Testing the model and your ik implementation 
- execution example
 ```shell
cd ravens
CUDA_VISIBLE_DEVICES=-1 python ravens/test.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion-easy --agent=transporter --n_demos=1000 --n_steps=20000# No need to use GPU


Loading pre-trained model at 20000 iterations.
============================ Task 3 : Transporter Network ============================

Test: 1/10
WARNING:tensorflow:From /home/timmy/pdm-f23/hw4/ravens/ravens/agents/transporter.py:167: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
Instructions for updating:
Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
W1205 20:59:22.550236 139775920095872 deprecation.py:323] From /home/timmy/pdm-f23/hw4/ravens/ravens/agents/transporter.py:167: set_learning_phase (from tensorflow.python.keras.backend) is deprecated and will be removed after 2020-10-11.
Instructions for updating:
Simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
Total Reward: 1.0 Done: True
Test: 2/10
Total Reward: 1.0 Done: True
Test: 3/10
Total Reward: 1.0 Done: True
Test: 4/10
Total Reward: 1.0 Done: True
Test: 5/10
Total Reward: 1.0 Done: True
Test: 6/10
Total Reward: 1.0 Done: True
Test: 7/10
Total Reward: 1.0 Done: True
Test: 8/10
Total Reward: 1.0 Done: True
Test: 9/10
Total Reward: 1.0 Done: True
Test: 10/10
Total Reward: 1.0 Done: True
====================================================================================
- Your Total Score : 10.000 / 10.000
====================================================================================
 ```
# Reference
https://github.com/google-research/ravens
