import numpy as np
import matplotlib.pyplot as plt

l1 = l2 = 1  
d = 0.05  
time = np.linspace(0, 2*np.pi, 1000)

y = np.cos(time)
y_dot = -np.sin(time)

def inverse_kinematics(x, y):
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = -np.arccos(cos_theta2) 
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    return theta1, theta2

theta1, theta2 = inverse_kinematics(d, y)

theta1_dot = np.gradient(theta1, time)
theta2_dot = np.gradient(theta2, time)

theta1_max = np.max(theta1)
max_theta1_time = time[np.argmax(theta1)]
theta2_at_max_theta1 = theta2[np.argmax(theta1)]

print(f"Part C Solution:\nMaximum theta1: {theta1_max} rad")
print(f"Time when theta1 is maximum: {max_theta1_time} seconds")
print(f"Corresponding theta2: {theta2_at_max_theta1} rad")

y_zero = 0
theta1_y_zero, theta2_y_zero = inverse_kinematics(d, y_zero)

print(f"\nValues when y = 0:")
print(f"Theta1: {theta1_y_zero} rad")
print(f"Theta2: {theta2_y_zero} rad")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(time, theta1, label=r'$\theta_1(t)$', color='b')
plt.plot(time, theta2, label=r'$\theta_2(t)$', color='r')
plt.title('Joint Angles vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time, theta1_dot, label=r'$\dot{\theta}_1(t)$', color='b')
plt.plot(time, theta2_dot, label=r'$\dot{\theta}_2(t)$', color='r')
plt.title('Joint Velocities vs Time')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Number4_with_PartC.png")
plt.show()
