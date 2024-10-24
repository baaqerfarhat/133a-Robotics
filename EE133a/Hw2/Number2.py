import numpy as np
import matplotlib.pyplot as plt


l1 = 3
l2 = 1
l3 = 0.5
time_start = 0
time_end = 10
num_points = 1000
t = np.linspace(time_start, time_end, num_points)

# Set a
w1_a = 1
w2_a = -3
w3_a = 9
phi1_a = 0
phi2_a = np.pi
phi3_a = np.pi / 2

# Set b
w1_b = 1
w2_b = -6
w3_b = 12
phi1_b = 0
phi2_b = np.pi
phi3_b = 0

# Set c
w1_c = 1
w2_c = -4
w3_c = 12
phi1_c = 0
phi2_c = np.pi
phi3_c = np.pi

def calculate_position_a():
    theta1_a = w1_a * t + phi1_a
    theta2_a = w2_a * t + phi2_a
    theta3_a = w3_a * t + phi3_a
    x_a = l1 * np.cos(theta1_a) + l2 * np.cos(theta1_a + theta2_a) + l3 * np.cos(theta1_a + theta2_a + theta3_a)
    y_a = l1 * np.sin(theta1_a) + l2 * np.sin(theta1_a + theta2_a) + l3 * np.sin(theta1_a + theta2_a + theta3_a)
    
    return x_a, y_a

def calculate_position_b():
    theta1_b = w1_b * t + phi1_b
    theta2_b = w2_b * t + phi2_b
    theta3_b = w3_b * t + phi3_b
    x_b = l1 * np.cos(theta1_b) + l2 * np.cos(theta1_b + theta2_b) + l3 * np.cos(theta1_b + theta2_b + theta3_b)
    y_b = l1 * np.sin(theta1_b) + l2 * np.sin(theta1_b + theta2_b) + l3 * np.sin(theta1_b + theta2_b + theta3_b)
    
    return x_b, y_b



def calculate_position_c():
    theta1_c = w1_c * t + phi1_c
    theta2_c = w2_c * t + phi2_c
    theta3_c = w3_c * t + phi3_c
    x_c = l1 * np.cos(theta1_c) + l2 * np.cos(theta1_c + theta2_c) + l3 * np.cos(theta1_c + theta2_c + theta3_c)
    y_c = l1 * np.sin(theta1_c) + l2 * np.sin(theta1_c + theta2_c) + l3 * np.sin(theta1_c + theta2_c + theta3_c)
    
    return x_c, y_c

x_a, y_a = calculate_position_a()
x_b, y_b = calculate_position_b()
x_c, y_c = calculate_position_c()

plt.figure(figsize=(15, 5))

# this is for Set a
plt.subplot(1, 3, 1)
plt.plot(x_a, y_a)
plt.title("Parh for set (a)")
plt.xlabel("X")
plt.ylabel("Y")

# this is for Set b
plt.subplot(1, 3, 2)
plt.plot(x_b, y_b)
plt.title("Parh for set (b)")
plt.xlabel("X")
plt.ylabel("Y")

# this is for Set c
plt.subplot(1, 3, 3)
plt.plot(x_c, y_c)
plt.title("Parh for set (c)")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.savefig("Number2Plot.png")
plt.show()
