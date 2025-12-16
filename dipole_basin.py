import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import g, mu_0, pi
#from tqdm import tqdm

# === 系统参数 ===
line_length = 0.4
hanging_point = np.array([0.0, 0.0, 0.5])
m = 0.1
damping_coefficient = 0.02
initial_speed = 0
velocity_threshold = 0.003

# === 磁铁参数 ===
pos_magnet_B = np.array([-0.05, 0.0, 0.0])
pos_magnet_D = np.array([0.05, 0.0, 0.0])
m_B = np.array([0.0, 0.0, 10.0])
m_D = np.array([0.0, 0.0, 10.0])

# === 磁偶极子力 ===
def rodrigues_rotation(v, k, theta):
        k = k / (np.linalg.norm(k) + 1e-12)
        return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

def rotated_magnetic_moment__direction(phi, m_dir):
        local_dl = np.array([-np.sin(phi), np.cos(phi), 0.0]) * radius_magnet_C
        default_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(m_dir, default_axis):
            return local_dl
        axis = np.cross(default_axis, m_dir)
        angle = np.arccos(np.clip(np.dot(default_axis, m_dir), -1.0, 1.0))
        return rodrigues_rotation(local_dl, axis, angle)
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

# === 判断函数 ===
def simulate_and_classify(x, y):
    if x**2 + y**2 >= line_length**2:
        return np.nan

    z = hanging_point[2] - np.sqrt(line_length**2 - x**2 - y**2)
    pos_init = np.array([x, y, z])
    vel_dir = np.cross(pos_init - hanging_point, [0, 0, 1])
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-8)
    vel_init = vel_dir * initial_speed
    y0 = np.concatenate([pos_init, vel_init])

    def motion_with_tension(t, y):
        pos = y[:3]
        vel = y[3:]
        radial = pos - hanging_point
        radial_unit = radial / np.linalg.norm(radial)
        m_C = radial_unit
        gravity = np.array([0, 0, -m * g])
        magnetic = dipole_force(m_B, m_C, pos_magnet_B, pos) + dipole_force(m_D, m_C, pos_magnet_D, pos)
        damping = -damping_coefficient * vel
        F_ext = gravity + magnetic + damping
        v2 = np.dot(vel, vel)
        tension = m * v2 / line_length + np.dot(F_ext, radial_unit)
        acc = (F_ext - tension * radial_unit) / m
        return np.concatenate([vel, acc])

    def stop(t, y):
        return np.linalg.norm(y[3:]) - velocity_threshold
    stop.terminal = True
    stop.direction = -1

    sol = solve_ivp(motion_with_tension, [0, 10], y0, events=stop, max_step=0.05)
    final_pos = sol.y[:3, -1]
    return 1 if np.linalg.norm(final_pos - pos_magnet_B) < np.linalg.norm(final_pos - pos_magnet_D) else 0

# === 扫描相图 ===
res = 10
x_vals = np.linspace(-0.2, 0.2, res)
y_vals = np.linspace(-0.2, 0.2, res)
Z = np.zeros((res, res))

for i, x in enumerate(x_vals):
    for j, y in enumerate(y_vals):
        Z[j, i] = simulate_and_classify(x, y)

# === 绘图 ===
plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[-0.2, 0.2, -0.2, 0.2], origin='lower', cmap='coolwarm', alpha=0.8)
plt.xlabel("Initial x (m)")
plt.ylabel("Initial y (m)")
plt.title("Attraction Region: 0 = D (Red), 1 = B (Blue)")
plt.colorbar(label="Final Magnet")
plt.grid(True)
plt.show()
