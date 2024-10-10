import numpy as np
import matplotlib.pyplot as plt

a = 4
b = 5
d = 6

save_path = r"C:\Users\farha\Desktop\Caltech\2024-2025\EE133a\Hw1"

def calculate_theta(phi, c):
    XB = d + c * np.cos(phi)
    YB = c * np.sin(phi)
    r = np.sqrt(XB**2 + YB**2)
    cos_beta = (a**2 + r**2 - b**2) / (2 * a * r)
    beta = np.arccos(np.clip(cos_beta, -1, 1)) 
    gamma = np.arctan2(YB, XB)
    theta_1 = gamma + beta
    theta_2 = gamma - beta
    theta_3 = theta_1 + 2 * np.pi
    theta_4 = theta_2 + 2 * np.pi
    
    return theta_1, theta_2, theta_3, theta_4

phi_values = np.linspace(-2*np.pi, 2*np.pi, 10000)
c_values = [2, 4, 5]

for c in c_values:
    theta_1_values = []
    theta_2_values = []
    theta_3_values = []
    theta_4_values = []
    
    for phi in phi_values:
        theta_1, theta_2, theta_3, theta_4 = calculate_theta(phi, c)
        theta_1_values.append(theta_1)
        theta_2_values.append(theta_2)
        theta_3_values.append(theta_3)
        theta_4_values.append(theta_4)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(phi_values, theta_1_values, label=f'', color='blue', s=1, marker='o')
    plt.scatter(phi_values, theta_2_values, label=f'', color='red', s=1, marker='o')
    plt.scatter(phi_values, theta_3_values, label=f'', color='green', s=1, marker='o')
    plt.scatter(phi_values, theta_4_values, label=f'', color='purple', s=1, marker='o')

    plt.xlabel('ϕ (radians)')
    plt.ylabel('θ (radians)')
    plt.title(f'Configuration Space of 4-bar Linkage for c = {c}m (ϕ vs θ)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{save_path}config_space_c_{c}m.png", dpi=300, bbox_inches='tight')

plt.show()
