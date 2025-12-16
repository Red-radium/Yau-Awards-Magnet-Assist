from scipy import integrate
import numpy as np
from scipy.constants import mu_0, pi, g
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

# 系统参数
m = 0.1  # 质量 (kg)
line_length = 0.4  # 绳长 (m)
hanging_point = np.array([0.0, 0.0, 0.5])  # 悬挂点

# 磁铁参数
radius_magnet = 0.1
length_magnet = 0.02
current_equiv = 1.0
pos_B1 = np.array([-0.05, 0.0, 0.0])
pos_B2 = np.array([0.05, 0.0, 0.0])

def compute_B_field_at(position, pos_ring):
    def biot_savart(phi, index):
        source = np.array([
            radius_magnet * np.cos(phi),
            radius_magnet * np.sin(phi),
            0.0
        ]) + pos_ring
        dl = np.array([
            -np.sin(phi),
            np.cos(phi),
            0.0
        ]) * radius_magnet
        r = position - source
        return np.cross(dl, r)[index] / np.linalg.norm(r)**3

    B = np.array([integrate.quad(biot_savart, 0, 2*pi, args=(i,))[0] for i in range(3)])
    return mu_0 * current_equiv / (4*pi) * B

def ring_force(pos_C, pos_ring):
    def dF(phi, index):
        dl = np.array([
            -np.sin(phi),
            np.cos(phi),
            0.0
        ]) * radius_magnet
        element = np.array([
            radius_magnet * np.cos(phi),
            radius_magnet * np.sin(phi),
            0.0
        ]) + pos_C
        B = compute_B_field_at(element, pos_ring)
        return np.cross(dl, B)[index]

    force = np.array([integrate.quad(dF, 0, 2*pi, args=(i,))[0] for i in range(3)])
    return force

def get_force(pos_C):
    F1 = ring_force(pos_C, pos_B1)
    F2 = ring_force(pos_C, pos_B2)
    return F1 + F2

def motion_eq(t, y):
    pos = y[:3]
    vel = y[3:]
    gravity = np.array([0, 0, -m * g])
    F_magnetic = get_force(pos)
    damping = -0.1 * vel
    F_ext = gravity + F_magnetic + damping

    radial = pos - hanging_point
    r_hat = radial / np.linalg.norm(radial)
    v2 = np.dot(vel, vel)
    radial_F = np.dot(F_ext, r_hat)
    tension = m * v2 / line_length + radial_F

    acc = (F_ext - tension * r_hat) / m
    return np.concatenate([vel, acc])

# 初始条件
x, y = 0.0, 0.1
z = hanging_point[2] - np.sqrt(line_length**2 - x**2 - y**2)
y0 = np.array([x, y, z, 0, 0, 0])
t_span = (0, 5)
t_eval = np.linspace(*t_span, 200)

with tqdm(total=len(t_eval), desc="Simulating") as pbar:
    def wrapped_motion_eq(t, y):
        pbar.update(1)
        return motion_eq(t, y)
    sol = solve_ivp(wrapped_motion_eq, t_span, y0, t_eval=t_eval)

# 绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], label='Trajectory')
ax.scatter(*hanging_point, c='red', label='Hanging Point')
ax.scatter(*pos_B1, c='blue', label='Magnet B1')
ax.scatter(*pos_B2, c='green', label='Magnet B2')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.title('3D Motion under Dual Magnetic Rings')
plt.show()
