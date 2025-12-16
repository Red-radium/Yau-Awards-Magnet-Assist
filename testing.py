import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 参数
b = 0.1
h = 0.5
magnets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
p = np.array([1, 1, 1, 1])

# 定义微分方程
def dynamics(t, y):
    pos = y[:2]
    vel = y[2:]
    force = -pos - b * vel
    for n in range(4):
        diff = magnets[n] - pos
        denom = (np.sum(diff**2) + h**2)**(2.5)
        force += p[n] * diff / denom
    return [vel[0], vel[1], force[0], force[1]]

# 网格
res = 3
x_range = np.linspace(-2, 2, res)
y_range = np.linspace(-2, 2, res)
result = np.zeros((res, res))

# 对每个网格点积分
for i, x0 in enumerate(x_range):
    for j, y0 in enumerate(y_range):
        sol = solve_ivp(dynamics, [0, 100], [x0, y0, 0, 0], max_step=0.1)
        final = sol.y[:2, -1]
        dists = np.linalg.norm(magnets - final[:, None].T, axis=1)
        result[j, i] = np.argmin(dists)

# 绘制结果
plt.imshow(result, extent=[-2, 2, -2, 2], origin='lower', cmap='tab10')
plt.title('Fractal Basins of Attraction')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
