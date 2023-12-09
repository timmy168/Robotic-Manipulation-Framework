import os, argparse, json
import numpy as np

from scipy.spatial.transform import Rotation as R

# for simulator
import pybullet as p

# for geometry information
from hw4_utils.bullet_utils import draw_coordinate, get_matrix_from_pose, get_pose_from_matrix, pose_7d_to_6d, pose_6d_to_7d

SIM_TIMESTEP = 1.0 / 240.0
JACOBIAN_SCORE_MAX = 10.0
JACOBIAN_ERROR_THRESH = 0.05
FK_SCORE_MAX = 10.0
FK_ERROR_THRESH = 0.005
TASK1_SCORE_MAX = JACOBIAN_SCORE_MAX + FK_SCORE_MAX

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    return np.cross(a, b)

def get_ur5_DH_params():
    dh_params = [
                    {'a':  0,      'd': 0.0892,  'alpha':  np.pi/2,  },    # joint1
                    {'a':  -0.425, 'd': 0,       'alpha':  0         },    # joint2
                    {'a':  -0.392, 'd': 0,       'alpha':  0         },    # joint3
                    {'a':  0.,     'd': 0.1093,  'alpha':  np.pi/2   },    # joint4
                    {'a':  0.,     'd': 0.09475, 'alpha': -np.pi/2   },    # joint5
                    {'a':  0,      'd': 0.2023,  'alpha': 0          },    # joint6
                ]
    return dh_params

def your_fk(DH_params : dict, q : list or tuple or np.ndarray, base_pos) -> np.ndarray:
    # robot initial position [0,0,0,0,0,0]
    base_pose = list(base_pos) + [0, 0, 0]  

    assert len(DH_params) == 6 and len(q) == 6, f'Both DH_params and q should contain 6 values,\n' \
                                                f'but get len(DH_params) = {DH_params}, len(q) = {len(q)}'
    # initialize
    A = get_matrix_from_pose(base_pose) # a 4x4 matrix, type should be np.ndarray
    print(A)
    jacobian = np.zeros((6, 6)) # a 6x6 matrix, type should be np.ndarray

    # foward kinematics
    T6 = np.eye(4)
    transforms = []
    for i, parameters in enumerate(DH_params):
        a, d, alpha = parameters['a'], parameters['d'], parameters['alpha']
        theta = q[i]
        # transform matrix
        transform = np.array([[np.cos(theta),-np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha),a*np.cos(theta)],
                              [np.sin(theta), np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
                              [0, np.sin(alpha), np.cos(alpha), d],
                              [0, 0, 0, 1]])
        print("Transform from joint"+str(i)+"to"+str(i+1),transform)
        A = A.dot(transform)
        T6 = T6.dot(transform)
        transforms.append(transform)
    
    # foward jacobian
    # hint : 
    # https://automaticaddison.com/the-ultimate-guide-to-jacobian-matrices-for-robotics/

    # adjustment don't touch
    adjustment = np.asarray([[ 0, -1,  0],
                             [ 0,  0,  0],
                             [ 0,  0, -1]])
    A[:3, :3] = A[:3,:3]@adjustment
    pose_7d = np.asarray(get_pose_from_matrix(A,7))

    return pose_7d, jacobian

if __name__=="__main__":
    DH_params = get_ur5_DH_params()
    q = [np.pi/2,np.pi/2,np.pi/2,np.pi/2,np.pi/2,np.pi/2]
    base_pos = [0,0,0]
    pose_7d, jacobian = your_fk(DH_params,q,base_pos)
    print(pose_7d)