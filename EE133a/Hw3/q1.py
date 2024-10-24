import numpy as np

theta_pan = 1.0  
theta_tilt = 0.5  
theta_roll = -0.2  

def rotation_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotation_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotation_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

R_pan = rotation_z(theta_pan)
R_tilt = rotation_y(theta_tilt)
R_roll = rotation_x(theta_roll)

R = np.dot(np.dot(R_pan, R_tilt), R_roll)
print("Rotation matrix for part (a):")
print(np.round(R, 4))

# Part b
R_given = np.array([[0.8839, 0.3536, -0.3062],
                    [-0.1768, -0.3536, -0.9186],
                    [-0.4330, 0.8660, -0.2500]])

def extract_pan_tilt_roll(R):
    theta_tilt = np.arcsin(R[2, 1])
    theta_pan = np.arctan2(R[1, 1], R[0, 1])
    theta_roll = np.arctan2(-R[2, 0], R[2, 2])
    return theta_pan, theta_tilt, theta_roll

theta_pan_solution1, theta_tilt_solution1, theta_roll_solution1 = extract_pan_tilt_roll(R_given)

print("\nSolution 1 for part (b):")
print(f"Pan angle: {theta_pan_solution1:.4f} rad")
print(f"Tilt angle: {theta_tilt_solution1:.4f} rad")
print(f"Roll angle: {theta_roll_solution1:.4f} rad")

theta_tilt_solution2 = np.pi - theta_tilt_solution1
print(f"Tilt angle (solution 2): {theta_tilt_solution2:.4f} rad")
