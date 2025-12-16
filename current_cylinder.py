from scipy import integrate
import numpy as np
from scipy.constants import mu_0, pi, g
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== 常数定义 =====
radius_magnet_A = 0.1
length_magnet_A = 0.02
current_equiv_A = 1.0
pos_magnet_A = np.array([-0.05, 0.0, 0.0])

radius_magnet_B = 0.1
length_magnet_B = 0.02
current_equiv_B = 1.0
pos_magnet_B = np.array([0.05, 0.0, 0.0])

radius_magnet_C = 0.1
length_magnet_C = 0.02
current_equiv_C = 1.0

hanging_point = np.array([0.0, 0.0, 0.5])
line_length = 0.2
m = 0.1

L0 = np.array([0.0, 0.0, 0.0])              # 初始角动量
m_dir0 = np.array([0.0, 0.0, 1.0])          # 初始磁矩方向：竖直向上

line_length = 0.4
hanging_point = np.array([0.0, 0.0, 0.5])
m = 0.1
damping_coefficient = 0.02
initial_speed = 0
velocity_threshold = 0.003

def simulate_and_classify(x, y):
    if x**2 + y**2 >= line_length**2:
        return np.nan
    z = hanging_point[2] - np.sqrt(line_length**2 - x**2 - y**2)
    pos_init = np.array([x, y, z])
    vel_dir = np.cross(pos_init - hanging_point, [0, 0, 1])
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-8)
    vel_init = vel_dir * initial_speed
    y0 = np.concatenate([pos_init, vel_init])

    def rodrigues_rotation(v, k, theta):
        k = k / (np.linalg.norm(k) + 1e-12)
        return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

    def rotated_current_element_direction(phi, m_dir):
        local_dl = np.array([-np.sin(phi), np.cos(phi), 0.0]) * radius_magnet_C
        default_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(m_dir, default_axis):
            return local_dl
        axis = np.cross(default_axis, m_dir)
        angle = np.arccos(np.clip(np.dot(default_axis, m_dir), -1.0, 1.0))
        return rodrigues_rotation(local_dl, axis, angle)

    def biot_savart_contribution_B(phi, field_eval_position, index):
        source_position = np.array([
            radius_magnet_B * np.cos(phi),
            radius_magnet_B * np.sin(phi),
            0.0
        ]+pos_magnet_B)
        l_element = np.array([-np.sin(phi), np.cos(phi), 0.0]) * radius_magnet_B
        displacement = field_eval_position - source_position
        r_norm = np.linalg.norm(displacement)
        return np.cross(l_element, displacement)[index] / r_norm**3

    def biot_savart_contribution_A(phi, field_eval_position, index):
        source_position = np.array([
            radius_magnet_A * np.cos(phi),
            radius_magnet_A * np.sin(phi),
            0.0
        ]+pos_magnet_A)
        l_element = np.array([-np.sin(phi), np.cos(phi), 0.0]) * radius_magnet_A
        displacement = field_eval_position - source_position
        r_norm = np.linalg.norm(displacement)
        return np.cross(l_element, displacement)[index] / r_norm**3

    def compute_B_field_at(position):
        B_result_B = np.array([
            integrate.quad(biot_savart_contribution_B, 0, 2*pi, args=(position, i))[0] for i in range(3)
        ])
        B_result_A = np.array([
            integrate.quad(biot_savart_contribution_B, 0, 2*pi, args=(position, i))[0] for i in range(3)
        ])
        return (B_result_A + B_result_B) * mu_0 * current_equiv_B / (4 * pi)

    def get_force_and_torque(pos_magnet_C, m_dir):
        def force_calculation(phi, index):
            dl = rotated_current_element_direction(phi, m_dir)
            pos = np.array([
                radius_magnet_C * np.cos(phi),
                radius_magnet_C * np.sin(phi),
                0.0
            ]) + pos_magnet_C
            B = compute_B_field_at(pos)
            return (current_equiv_C * np.cross(dl, B))[index]

        def torque_calculation(phi, index):
            dl = rotated_current_element_direction(phi, m_dir)
            pos = np.array([
                radius_magnet_C * np.cos(phi),
                radius_magnet_C * np.sin(phi),
                0.0
            ]) + pos_magnet_C
            B = compute_B_field_at(pos)
            dF = current_equiv_C * np.cross(dl, B)
            r = pos - pos_magnet_C
            return np.cross(r, dF)[index]

        force = np.array([integrate.quad(force_calculation, 0, 2*pi, args=(i,))[0] for i in range(3)])
        torque = np.array([integrate.quad(torque_calculation, 0, 2*pi, args=(i,))[0] for i in range(3)])
        return force, torque

    def motion_with_rotation(t, y):
        pos = y[0:3]
        vel = y[3:6]
        m_dir = y[6:9]
        L = y[9:12]

        gravity = np.array([0, 0, -m * g])
        force, torque = get_force_and_torque(pos, m_dir)
        damping = -0.1 * vel
        F_ext = gravity + force + damping

        radial = pos - hanging_point
        radial_unit = radial / np.linalg.norm(radial)
        v_squared = np.dot(vel, vel)
        radial_force = np.dot(F_ext, radial_unit)
        tension = m * v_squared / line_length + radial_force
        F_total = F_ext - tension * radial_unit
        acc = F_total / m

        I = 0.5 * m * radius_magnet_C**2
        omega = L / I
        m_dir_dot = np.cross(omega, m_dir)
        L_dot = torque

        return np.concatenate([vel, acc, m_dir_dot, L_dot])

    # ===== 初始条件 =====
    x, y = 0.0, 0.1
    z = hanging_point[2] - np.sqrt(line_length**2 - x**2 - y**2)
    pos_init = np.array([x, y, z])
    vel_init = np.array([0.0, 0.0, 0.0])
    y0 = np.concatenate([pos_init, vel_init, m_dir0, L0])
    t_span = (0, 5)
    t_eval = np.linspace(*t_span, 200)

    # ===== 积分求解 =====
    with tqdm(total=len(t_eval), desc="Simulating") as pbar:
        def wrapped_motion(t, y):
            pbar.update(1)
            return motion_with_rotation(t, y)

        sol = solve_ivp(wrapped_motion, t_span, y0, t_eval=t_eval)

    def stop(t, y):
        return np.linalg.norm(y[3:]) - velocity_threshold
    stop.terminal = True
    stop.direction = -1

res = 10
x_vals = np.linspace(-0.2, 0.2, res)
y_vals = np.linspace(-0.2, 0.2, res)
Z = np.zeros((res, res))

for i, x in enumerate(tqdm(x_vals, desc="Simulating grid")):
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
