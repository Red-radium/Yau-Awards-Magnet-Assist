from scipy import integrate
import numpy as np
from scipy.constants import mu_0, pi, g
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pickle


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

# ===== 绘图（可选）=====

# 2. 动力学方程定义 (示例：磁摆运动)
def motion_equation(t, y):
    x, y, z, vx, vy, vz = y
    # 示例方程 (替换为您的实际方程)
    ax = -x + 0.1*vy - 0.05*vx
    ay = -y - 0.1*vx - 0.05*vy
    az = -z - 0.05*vz
    return [vx, vy, vz, ax, ay, az]

# 3. 运行模拟并保存结果
def run_simulation():
    print("Running simulation...")
    with tqdm(total=len(t_eval), desc="Progress") as pbar:
        def progress_wrapper(t, y):
            pbar.update(1)
            return motion_equation(t, y)
        
        sol = solve_ivp(progress_wrapper, t_span, y0, t_eval=t_eval, method='RK45')
    
    results = {
        'time': sol.t,
        'position': sol.y[:3].T,  # (N,3) 数组
        'velocity': sol.y[3:].T,  # (N,3) 数组
        'parameters': {
            't_span': t_span,
            'initial_state': y0
        }
    }
    with open('trajectory_data.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results

# 4. 绘图函数
def plot_xt_yt(results):
    """绘制x-t和y-t图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # x-t图
    ax1.plot(results['time'], results['position'][:,0], 'r-')
    ax1.set_ylabel('x position (m)')
    ax1.grid(True)
    
    # y-t图
    ax2.plot(results['time'], results['position'][:,1], 'b-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('y position (m)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('xt_yt_plots.png')
    plt.show()

def create_xy_animation(results):
    """创建x-y平面轨迹动画"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(results['position'][:,0].min()-0.1, results['position'][:,0].max()+0.1)
    ax.set_ylim(results['position'][:,1].min()-0.1, results['position'][:,1].max()+0.1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True)
    
    line, = ax.plot([], [], 'b-', lw=1)
    point, = ax.plot([], [], 'ro', ms=6)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text('')
        return line, point, time_text
    
    def update(frame):
        line.set_data(results['position'][:frame,0], results['position'][:frame,1])
        point.set_data(results['position'][frame,0], results['position'][frame,1])
        time_text.set_text(f'Time = {results["time"][frame]:.2f}s')
        return line, point, time_text
    
    ani = FuncAnimation(fig, update, frames=len(results['time']),
                        init_func=init, blit=True, interval=20)
    ani.save('xy_trajectory.gif', writer='pillow', fps=30)
    plt.close()
    return ani

def create_3d_animation(results):
    """创建3D轨迹动画"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    pos = results['position']
    ax.set_xlim(pos[:,0].min()-0.1, pos[:,0].max()+0.1)
    ax.set_ylim(pos[:,1].min()-0.1, pos[:,1].max()+0.1)
    ax.set_zlim(pos[:,2].min()-0.1, pos[:,2].max()+0.1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    
    line, = ax.plot([], [], [], 'b-', lw=1)
    point, = ax.plot([], [], [], 'ro', ms=6)
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        time_text.set_text('')
        return line, point, time_text
    
    def update(frame):
        line.set_data(pos[:frame,0], pos[:frame,1])
        line.set_3d_properties(pos[:frame,2])
        point.set_data(pos[frame,0], pos[frame,1])
        point.set_3d_properties(pos[frame,2])
        time_text.set_text(f'Time = {results["time"][frame]:.2f}s')
        return line, point, time_text
    
    ani = FuncAnimation(fig, update, frames=len(pos),
                        init_func=init, blit=True, interval=20)
    ani.save('3d_trajectory.gif', writer='pillow', fps=30)
    plt.close()
    return ani

# 5. 主程序
if __name__ == "__main__":
    # 运行或加载模拟
    try:
        with open('trajectory_data.pkl', 'rb') as f:
            results = pickle.load(f)
        print("Loaded existing simulation data")
    except FileNotFoundError:
        results = run_simulation()
    
    # 绘制静态图
    plot_xt_yt(results)
    
    # 创建动画 (会在后台生成并保存为gif)
    print("Creating xy trajectory animation...")
    create_xy_animation(results)
    
    print("Creating 3D trajectory animation...")
    create_3d_animation(results)
    
    print("All plots saved as:")
    print("- xt_yt_plots.png")
    print("- xy_trajectory.gif")
    print("- 3d_trajectory.gif")