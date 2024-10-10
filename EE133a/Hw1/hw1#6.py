import numpy as np
import matplotlib.pyplot as plt

l1 = 5
l2 = 1
l3 = 2
def calculate_workspace(theta1_range, theta2_range, theta3_range, num_points=25):
    theta1_values = np.linspace(theta1_range[0], theta1_range[1], num_points)
    theta2_values = np.linspace(theta2_range[0], theta2_range[1], num_points)
    theta3_values = np.linspace(theta3_range[0], theta3_range[1], num_points)
    
    X = []
    Y = []
    for theta1 in theta1_values:
        for theta2 in theta2_values:
            for theta3 in theta3_values:
                # Forward kinematics
                x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
                y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)                
                if len(X) > 0 and np.sqrt((x - X[-1])**2 + (y - Y[-1])**2) > 1.0:
                    X.append(None)  
                    Y.append(None)  
                
                X.append(x)
                Y.append(y)
    
    return np.array(X), np.array(Y)

def plot_and_save_workspace(X, Y, title, filename, color='blue'):
    plt.figure(figsize=(6, 6))
    plt.plot(X, Y, color=color, linewidth=0.5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.show()

theta1_range_a = [0, 2*np.pi]
theta2_range_a = [0, 2*np.pi]
theta3_range_a = [0, 2*np.pi]
X_a, Y_a = calculate_workspace(theta1_range_a, theta2_range_a, theta3_range_a)

theta1_range_b = [0, np.pi]
theta2_range_b = [0, 2*np.pi]
theta3_range_b = [0, 2*np.pi]
X_b, Y_b = calculate_workspace(theta1_range_b, theta2_range_b, theta3_range_b)

theta1_range_c = [0, np.pi]
theta2_range_c = [0, np.pi]
theta3_range_c = [0, np.pi]
X_c, Y_c = calculate_workspace(theta1_range_c, theta2_range_c, theta3_range_c)

plot_and_save_workspace(X_a, Y_a, 'Workspace (a): All angles unlimited', 'workspace_case_a', color='blue')
plot_and_save_workspace(X_b, Y_b, 'Workspace (b): θ1 limited to [0, π]', 'workspace_case_b', color='green')
plot_and_save_workspace(X_c, Y_c, 'Workspace (c): All angles limited to [0, π]', 'workspace_case_c', color='red')
