import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.constants import g, mu_0, pi

# === 系统参数 ===
line_length = 0.49  # 绳长 (m)
hanging_point = np.array([0.0, 0.0, 0.5])  # 绳子上端位置
m = 0.1  # 磁铁质量 (kg)

# === 初始位置与初速度（用户可自定义）===
x, y = 0, 0.1
z = hanging_point[2] - np.sqrt(line_length**2 - x**2 - y**2)
pos_magnet_C = np.array([x, y, z])
initial_velocity = np.array([1, 0, 0.0])  # <<< 修改此行控制初始速度
y0 = np.concatenate([pos_magnet_C, initial_velocity])

# === 磁偶极子参数 ===
pos_magnet_B = np.array([0.0, 0.0, 0.0])      # 固定磁铁位置
m_B = np.array([0.0, 0.0, -10.0])                # 固定磁铁磁矩方向
m_C = np.array([0.0, 0.0, 10.0])                # 移动磁铁磁矩方向

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

# === 牛顿第二定律 + 拉力约束 ===
def motion_with_tension(t, y):
    pos = y[:3]
    vel = y[3:]

    gravity = np.array([0, 0, -m * g])
    magnetic_force = dipole_force(m_B, m_C, pos_magnet_B, pos)
    damping = -0.1 * vel  # 微小阻尼防止发散，可设为0关闭

    # 总外力（不含拉力）
    F_ext = gravity + magnetic_force + damping

    # 绳子方向与单位向量
    radial = pos - hanging_point
    radial_unit = radial / np.linalg.norm(radial)

    # 拉力计算：平衡外力在径向的投影 + 向心力
    v_squared = np.dot(vel, vel)
    radial_force = np.dot(F_ext, radial_unit)
    tension = m * v_squared / line_length + radial_force

    # 总加速度：总外力 + 拉力反作用
    F_total = F_ext - tension * radial_unit
    acc = F_total / m

    return np.concatenate([vel, acc])

'''# === 积分求解 ===
t_span = (0, 20)
t_eval = np.linspace(*t_span, 4000)
sol = solve_ivp(motion_with_tension, t_span, y0, t_eval=t_eval)

# === 动图绘制 ===
from renderer import *
render_trajectory(sol,[(0,0)],speed=10)
render_d_t(sol,1)'''
