import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.constants import g, mu_0, pi

# === 系统参数 ===
line_length = 0.49  # 绳长 (m)
hanging_point = np.array([0.0, 0.0, 0.5])  # 绳子上端位置
m = 0.1  # 磁铁质量 (kg)

# === 初始位置与初速度 ===
x, y = 0.0, 0.1
z = hanging_point[2] - np.sqrt(line_length**2 - x**2 - y**2)
pos_magnet_C = np.array([x, y, z])
initial_velocity = np.array([1, 0.0, 0.0])
y0 = np.concatenate([pos_magnet_C, initial_velocity])

# === 固定磁铁 B 和 D 的位置与磁矩 ===
pos_magnet_B = np.array([-0.1, 0.0, 0.0])
pos_magnet_D = np.array([0.1, 0.0, 0.0])
m_B = np.array([0.0, 0.0, -10.0])
m_D = np.array([0.0, 0.0, -10.0])

# === 磁偶极子之间的力 ===
def dipole_force(m1, m2, r1, r2):
    R = r2 - r1
    R_mag = np.linalg.norm(R)
    if R_mag == 0:
        return np.zeros(3)
    R_hat = R / R_mag
    mu = mu_0 / (4 * pi)
    term1 = np.dot(m2, R_hat) * m1
    term2 = np.dot(m1, m2) * R_hat
    term3 = np.dot(m1, R_hat) * m2
    term4 = 5 * np.dot(m1, R_hat) * np.dot(m2, R_hat) * R_hat
    return (3 * mu / R_mag**4) * (term1 + term2 + term3 - term4)

# === 动力学方程：考虑拉力和动态磁矩方向 ===
def motion_with_tension(t, y):
    pos = y[:3]
    vel = y[3:]

    # 动态磁矩方向（沿着从悬挂点指向当前位置的单位向量）
    radial = pos - hanging_point
    radial_unit = radial / np.linalg.norm(radial)
    m_C = radial_unit

    gravity = np.array([0, 0, -m * g])
    magnetic_force = (
        dipole_force(m_B, m_C, pos_magnet_B, pos) +
        dipole_force(m_D, m_C, pos_magnet_D, pos)
    )
    damping = -0.05 * vel
    F_ext = gravity + magnetic_force + damping

    # 拉力补偿径向分量 + 向心力
    v_squared = np.dot(vel, vel)
    radial_force = np.dot(F_ext, radial_unit)
    tension = m * v_squared / line_length + radial_force
    F_total = F_ext - tension * radial_unit
    acc = F_total / m

    return np.concatenate([vel, acc])

# === 数值积分 ===
t_span = (0, 10)
t_eval = np.linspace(*t_span, 400)
sol = solve_ivp(motion_with_tension, t_span, y0, t_eval=t_eval)

# === 动图绘制 ===
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], lw=2)
point, = ax.plot([], [], [], 'ro')
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.25, 0.25)
ax.set_zlim(0.1, 0.5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Trajectory with Dynamic Magnetic Moment')

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

def update(frame):
    line.set_data(sol.y[0][:frame], sol.y[1][:frame])
    line.set_3d_properties(sol.y[2][:frame])
    point.set_data([sol.y[0][frame]], [sol.y[1][frame]])
    point.set_3d_properties([sol.y[2][frame]])
    return line, point

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)
plt.show()
