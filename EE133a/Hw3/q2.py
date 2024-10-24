import numpy as np

def rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def rot_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def trans_y(L):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, L],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def trans_z(L):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, L],
        [0, 0, 0, 1]
    ])

theta_1 = -np.pi / 4
theta_2 = np.pi / 3
theta_3 = np.pi / 3
theta_4 = np.pi / 4
L = 0.4

T_0_1 = np.dot(trans_z(L), rot_z(theta_1))
T_1_2 = np.dot(trans_y(L), rot_y(theta_2))
T_2_3 = np.dot(trans_z(-L), rot_z(-theta_3))
T_3_4 = np.dot(trans_y(L), rot_y(theta_4))
T_0_t = np.dot(T_0_1, np.dot(T_1_2, np.dot(T_2_3, T_3_4)))

print(T_0_t)
