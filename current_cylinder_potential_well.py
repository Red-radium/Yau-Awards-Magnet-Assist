import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0, pi, g
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from matplotlib import ticker
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import tqdm

# 参数设置
I = 200.0       # 环电流 (A)
R = 0.05        # 环半径 (m)
l = 0.2         # 摆长 (m)
m = 0.01         # 质量 (kg)
m_mag = 1.0     # 磁矩 (A·m²)
z_height = 0.2  # 观测高度 (m)
h = 0.05        # 挂点高度 (m)

# 创建网格
x = np.linspace(-0.15, 0.15, 30)
y = np.linspace(-0.15, 0.15, 30)
X, Y = np.meshgrid(x, y)

# 1. 计算重力势能 U_gravity = mgl(1 - cosθ)
theta = np.arccos((h*h - np.sqrt(l**2 - X**2 - Y**2))/l)
U_gravity = m * g * l * (np.cos(theta)-1)

# 2. 计算磁势能 (保持原有计算方式不变)
def Bz_integrand(phi, x, y):
    dx = R*np.cos(phi) - x
    dy = R*np.sin(phi) - y
    r = np.sqrt(dx**2 + dy**2 + z_height**2)
    return (mu_0*I*R*z_height)/(4*pi*r**3) * (1 + (R - x*np.cos(phi) - y*np.sin(phi))/r**2)

def Bz(x, y):
    return quad(lambda phi: Bz_integrand(phi, x, y), 0, 2*pi)[0]

B = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        B[i,j] = Bz(X[i,j], Y[i,j])
U_magnetic = -m_mag * B  # 磁势能 U = -m·B

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

# 图1: 重力势能
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Y, U_gravity, cmap='Blues', alpha=0.8)
format_axis(ax1, 'Gravitational Potential\n$U = mgl(1 - \\cos\\theta)$')

# 图2: 磁势能
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_magnetic, cmap='Reds', alpha=0.8)
format_axis(ax2, 'Magnetic Potential\n$U = -\\mathbf{m}\\cdot\\mathbf{B}$')

# 图3: 联合势能
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(X, Y, U_total, cmap='viridis', alpha=0.8)
format_axis(ax3, 'Combined Potential\n$U_{total} = U_{gravity} + U_{magnetic}$')

# 调整布局和colorbar
plt.subplots_adjust(wspace=0.5, right=0.88)
cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
fig.colorbar(surf3, cax=cbar_ax, label='Energy (J)')

plt.show()