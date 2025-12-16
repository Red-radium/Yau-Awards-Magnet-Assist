import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0, pi, g
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker

# 参数设置
I = 300.0       # 环电流 (A)
R = 0.05        # 环半径 (m)
l = 0.40         # 摆长 (m)
m = 0.03        # 质量 (kg)
m_mag = 5.0     # 磁矩 (A·m²)
z_height = 0.2  # 观测高度 (m)
h = 0.45         # 挂点高度 (m)

# 创建网格
x = np.linspace(-0.15, 0.15, 30)
y = np.linspace(-0.15, 0.15, 30)
X, Y = np.meshgrid(x, y)

# 1. 计算重力势能 U_gravity = mgl(1 - cosθ)
theta = np.arccos((h - np.sqrt(l**2 - X**2 - Y**2))/l)
U_gravity = m * g * l * (np.cos(theta)-1)

# 2. 磁偶极子模型计算磁势能 (替代原电流环精确解)
def dipole_potential(x, y, z):
    """磁偶极子势能公式: U = -m·B = (μ0/4π) [3(m·r̂)r̂ - m]/r³"""
    r1 = np.sqrt((x-0.07)**2 + y**2 + z**2)
    r2 = np.sqrt((x+0.07)**2 + y**2 + z**2)
    m_dipole = np.array([0, 0, m_mag])  # 假设磁矩沿z轴
    if r1 < 1e-10 and r2 < 1e-10:  # 避免除以零
        return 0
    r_hat1 = np.array([x-0.07, y, z]) / r1
    r_hat2 = np.array([x+0.07, y, z]) / r2
    term1 = 3 * np.dot(m_dipole, r_hat1) * r_hat1 - m_dipole
    term2 = 3 * np.dot(m_dipole, r_hat2) * r_hat2 - m_dipole
    potential1= -(mu_0/(4*pi)) * np.dot(m_dipole, term1) / r1**3
    potential2= -(mu_0/(4*pi)) * np.dot(m_dipole, term2) / r2**3
    return potential1 + potential2

# 计算磁势能网格
U_magnetic = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        # 计算摆球位置 (z = h - sqrt(l² - x² - y²))
        z = h - np.sqrt(l**2 - X[i,j]**2 - Y[i,j]**2)
        U_magnetic[i,j] = dipole_potential(X[i,j], Y[i,j], z)

# 3. 联合势能
U_total = U_gravity + U_magnetic


# ===== 绘图设置 =====
fig = plt.figure(figsize=(18, 6))

# 通用格式设置
def format_axis(ax, title):
    ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax.zaxis.labelpad = 20
    ax.tick_params(axis='z', pad=10, direction='out')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Energy (J)')
    ax.set_title(title)
    ax.xaxis._axinfo['grid']['linestyle'] = "--"
    ax.yaxis._axinfo['grid']['linestyle'] = "--"
    ax.zaxis._axinfo['grid']['linestyle'] = "--"
    ax.xaxis._axinfo['grid']['alpha'] = 0.3
    ax.yaxis._axinfo['grid']['alpha'] = 0.3
    ax.zaxis._axinfo['grid']['alpha'] = 0.3

# 图1: 重力势能 (保持不变)
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Y, U_gravity, cmap='Blues', alpha=0.8)
format_axis(ax1, 'Gravitational Potential')

# 图2: 磁势能 (偶极子近似)
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_magnetic, cmap='Reds', alpha=0.8)
format_axis(ax2, 'Magnetic Potential')

# 图3: 联合势能
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X, Y, U_total, cmap='viridis', alpha=0.8)
format_axis(ax3, 'Total Potential')

# 调整布局和colorbar
plt.subplots_adjust(wspace=0.5, right=0.88)
# 调整colorbar位置和大小
cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(surf3, cax=cbar_ax, label='Energy (J)')


plt.show()