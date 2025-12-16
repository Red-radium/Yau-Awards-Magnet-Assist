import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import mu_0, pi
from matplotlib.patches import Patch

# 参数设置
m_dipole = 1.0    # 磁偶极矩 (A·m²)
I_loop = 127.0    # 环电流 (A)
R = 0.05          # 环半径 (m)
z_min = 0.05      # 误差分析起始高度

# 创建网格
x = np.linspace(-0.5, 0.5, 100)
z = np.linspace(-0.5, 0.5, 100)
X, Z = np.meshgrid(x, z)
r = np.sqrt(X**2 + Z**2)

# 计算磁场 (矢量形式)
B_dipole = (mu_0/(4*pi)) * (3*(m_dipole*Z)*Z/r**5 - m_dipole/r**3)  # 偶极子场
B_loop = (mu_0*I_loop*R**2)/(2*(R**2 + r**2)**(1.5))               # 环电流场

# ==============================================
# 第一个图：3D磁场对比图（带图例）
# ==============================================
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# 绘制偶极子场（蓝色半透明表面）
dipole_surf = ax1.plot_surface(X, Z, np.abs(B_dipole), 
                    color='blue',
                    alpha=0.6,
                    edgecolor='navy',
                    linewidth=0.3,
                    label='Dipole Field')

# 绘制环电流场（红色线框）
loop_surf = ax1.plot_wireframe(X, Z, np.abs(B_loop), 
                    color='red',
                    linewidth=1.5,
                    alpha=0.8,
                    label='Current Loop Field')

# 创建自定义图例（3D图需要特殊处理）
legend_elements = [
    Patch(facecolor='blue', alpha=0.6, edgecolor='navy', label='Dipole Field'),
    Patch(facecolor='red', alpha=0.8, edgecolor='red', label='Current Cylinder Field')
]
ax1.legend(handles=legend_elements, loc='upper right')

ax1.set_title('3D Magnetic Field Comparison', pad=20)
ax1.set_xlabel('x (m)', labelpad=10)
ax1.set_ylabel('z (m)', labelpad=10)
ax1.set_zlabel('|B| (T)', labelpad=10)
ax1.view_init(elev=30, azim=-45)

# 添加颜色条（示例：用偶极子场的数值范围）
mappable = plt.cm.ScalarMappable(cmap='cool')
mappable.set_array(np.abs(B_dipole))
cbar = fig1.colorbar(mappable, ax=ax1, shrink=0.6, pad=0.1)
cbar.set_label('Field Strength (T)')

plt.tight_layout()
plt.show()

# ==============================================
# 第二个图：误差剖面图（保持不变）
# ==============================================
# ...（保持原有代码不变）...

# ==============================================
# 第二个图：误差剖面图
# ==============================================
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)

# 仅绘制x=0中心线上的误差
z_selected = z[z >= z_min]
x_pos = 0.0  # 中心线
idx = np.argmin(np.abs(x - x_pos))
ax2.plot(z_selected, percent_error[z >= z_min, idx], 
         'b-', linewidth=2, 
         label=f'x={x_pos:.3f}m')

ax2.set_title(f'Vertical Error Distribution (z≥{z_min}m)')
ax2.set_xlabel('z (m)')
ax2.set_ylabel('Percentage Error (%)')
ax2.grid(True)
ax2.legend()
ax2.set_ylim(0, 100)  # 扩大y轴范围以容纳标注

plt.tight_layout()
plt.show()