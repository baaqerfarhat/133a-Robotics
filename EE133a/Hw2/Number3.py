import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 10, 1000)
theta_pan = (np.pi / 3) * np.sin(2 * time)
theta_tilt = (np.pi / 3) * np.sin(time) - (np.pi / 9) * np.cos(6 * time)

s_y = 2 
d_x = np.cos(theta_tilt) * np.sin(theta_pan)
d_y = np.cos(theta_tilt) * np.cos(theta_pan)
d_z = np.sin(theta_tilt)

s_x = (d_x / d_y) * s_y
s_z = (d_z / d_y) * s_y


plt.figure(figsize=(8, 6))
plt.plot(s_x, s_z, label="Laser Path", color='b')
plt.title('Laser Pointer Trajectory on the Screen (s_x vs s_z)')
plt.xlabel(r'$s_x(t)$ [m]')
plt.ylabel(r'$s_z(t)$ [m]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fixed3")
plt.show()
