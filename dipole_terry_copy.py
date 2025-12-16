from pathlib import Path
from scipy.constants import g, mu_0, pi
import numpy as np
#np.seterr(all='raise') # 所有warning raise成error

class Magnet: # 固定的磁铁
    def __init__(self,pos,ma): # ma 是磁矩，和m区分
        self.pos = np.array(pos)
        self.ma = np.array(ma)

def dipole_force(m1, m2, r1, r2):
    R = r2 - r1
    R_mag = np.linalg.norm(R)
    if R_mag == 0:
        print(0)
        return np.zeros(3)
    R_hat = R / R_mag
    mu = mu_0 / (4 * pi)
    term1 = np.dot(m2, R_hat) * m1
    term2 = np.dot(m1, m2) * R_hat
    term3 = np.dot(m1, R_hat) * m2
    term4 = 5 * np.dot(m1, R_hat) * np.dot(m2, R_hat) * R_hat
    #print(term1,term2,term3,term4)
    try:
        return (3 * mu / R_mag**4) * (term1 + term2 + term3 - term4)
    except:
        print(r1,r2)

class Experiment: # 整个实验
    def __init__(self,x=0.0,y=0.0,v0=(0.1,0.0,0.0),h = 0.34, l = 0.292, m = 0.003976, ma = 0.557, d = 0.0):
        # ma = 1.0
        self.line_length = l
        self.hanging_point = np.array([0.0,0.0,h])
        self.m = m
        self.pos = np.array([0.0,0.0,0.0])
        self.ma = np.array([0.0,0.0,ma])
        self.pos[0] = x
        self.pos[1] = y
        self.v = np.array(v0, dtype=float)
        if x**2 + y**2 > self.line_length:
            raise Exception(f"line length {self.line_length} is not enough to get ({x},{y})")
        else:
            self.pos[2] = self.hanging_point[2] - np.sqrt(self.line_length**2 - self.pos[0]**2 - self.pos[1]**2)
        #print(self.pos)
        self.magnets = [Magnet([d,0.0,0.0],[0.0,0.0,ma])]
        self.dt = 0.001 #0.01 # dt
    def motion_with_tension(self):
        pos = self.pos
        vel = self.v
        m = self.m
        damping_co = 0.0007

        gravity = np.array([0, 0, -m * g])
        magnetic_force = 0
        for magnet in self.magnets:
            magnetic_force += dipole_force(magnet.ma, self.ma, magnet.pos, self.pos)
        damping = -damping_co * vel

        F_ext = gravity + magnetic_force + damping

        radial = pos - self.hanging_point
        radial_unit = radial / np.linalg.norm(radial)

        v_squared = np.dot(vel, vel)
        radial_force = np.dot(F_ext, radial_unit)
        tension = m * v_squared / self.line_length + radial_force

        F_total = F_ext - tension * radial_unit
        if np.linalg.norm(F_total) > 1:
            pass
        acc = F_total / m

        return acc
    def simulation(self,x=0.0,y=0.0,v0=(0.1,0.0,0.0),max_time=30.0,terminate_acc=0.01,terminate_vel=0.02,
                  h=None, l=None, m=None, ma=None, d=None):
        # Re-initialize with optional overrides to sweep parameters.
        h_val = self.hanging_point[2] if hasattr(self, "hanging_point") else 0.34
        l_val = self.line_length if hasattr(self, "line_length") else 0.292
        m_val = self.m if hasattr(self, "m") else 0.003976
        ma_val = float(self.ma[2]) if hasattr(self, "ma") else 0.557
        d_val = self.magnets[0].pos[0] if hasattr(self, "magnets") else 0.0
        self.__init__(
            x,
            y,
            v0=v0,
            h=h if h is not None else h_val,
            l=l if l is not None else l_val,
            m=m if m is not None else m_val,
            ma=ma if ma is not None else ma_val,
            d=d if d is not None else d_val,
        )
        #print(self.pos)
        t = 0
        T = []
        Pos = []
        while t < max_time:
            acc = self.motion_with_tension()
            self.v += acc*self.dt
            self.pos += self.v*self.dt

            radial = self.pos - self.hanging_point
            rhat = radial / np.linalg.norm(radial)
            self.pos = self.hanging_point + rhat * self.line_length

            #    so v is purely tangential—no energy kick!
            self.v -= np.dot(self.v, rhat) * rhat

            t += self.dt
            T.append(t)
            Pos.append(self.pos.copy())
            if np.linalg.norm(acc) <= terminate_acc and np.linalg.norm(self.v) <= terminate_vel:
                #print(acc)
                #print(self.v)
                break
            #print(self.pos)
        return T,Pos



import matplotlib.pyplot as plt
from renderer import *


# ====== 分析：从 x(t), y(t) 提取瞬时频率并拟合 ======
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def load_experiment_data(path: Path):
    """Load t, x, y (and optional vx, vy) columns from a tab-delimited file with two-line header."""
    raw = np.genfromtxt(path, skip_header=2, delimiter="\t")
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    # Drop rows where x or y is NaN to avoid broken plots/peak detection.
    raw = raw[~np.isnan(raw[:, 1]) & ~np.isnan(raw[:, 2])]
    t = raw[:, 0]
    x = raw[:, 1]
    y = raw[:, 2]
    vx = raw[:, 3] if raw.shape[1] > 3 else None
    vy = raw[:, 4] if raw.shape[1] > 4 else None
    return t, x, y, vx, vy


def amplitude(x, y):
    return np.sqrt(x**2 + y**2)


def moving_average(arr, w=5):
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def freq_from_peaks(t, series, distance=100, smooth=5):
    """Return smoothed frequency curve + raw peak-based frequencies for a single series."""
    peaks, _ = find_peaks(series, distance=distance)
    if len(peaks) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    peak_times = t[peaks]
    period = np.diff(peak_times)
    freq = 1.0 / period
    freq_time = peak_times[1:]
    if len(freq) < smooth:
        return freq_time, freq, freq_time, freq
    freq_s = moving_average(freq, smooth)
    offset = (smooth - 1) // 2
    time_s = freq_time[offset : offset + len(freq_s)]
    return time_s, freq_s, freq_time, freq


def initial_velocity_from_data(t, x, y, vx, vy):
    """Pick initial velocity from file columns if present, otherwise finite-difference estimate."""
    # if vx is not None and vy is not None and np.isfinite(vx[0]) and np.isfinite(vy[0]):
    return float(vx[0]), float(vy[0])
    # if len(t) >= 2:
    #     dt = t[1] - t[0]
    #     if dt != 0:
    #         return float((x[1] - x[0]) / dt), float((y[1] - y[0]) / dt)
    # return 0.0, 0.0


def max_freq_for_run(ma, l, m, x_fraction=0.08, v0x=0.1):
    """Simulate once with the given parameters and return max instantaneous frequency."""
    x0 = x_fraction * l
    y0 = 0.0
    sim_T, sim_Pos = Experiment().simulation(x0, y0, v0=(v0x, 0.0, 0.0), l=l, m=m, ma=ma)
    if not len(sim_T):
        return np.nan
    sim_T = np.array(sim_T)
    sim_Pos = np.array(sim_Pos)
    amp = amplitude(sim_Pos[:, 0], sim_Pos[:, 1])
    _, _, _, freq_raw = freq_from_peaks(sim_T, amp, distance=8, smooth=3)
    return np.nanmax(freq_raw) if len(freq_raw) else np.nan

if __name__ == "__main__":
    # ----------- Load experimental data and run a matching simulation -----------
    data_file = Path("data/32.6cm高度26.3摆长15_3磁铁4.3间距数据/1.txt")
    exp_t, exp_x, exp_y, exp_vx, exp_vy = load_experiment_data(data_file)

    r = 1.5
    height = 0.3
    m = np.pi*r**2*height*7.5
    e = Experiment()

    exp_t = np.array(exp_t)
    exp_x = np.array(exp_x) * 29.7/34
    exp_y = np.array(exp_y) * 29.7/34
    if exp_vx is not None:
        exp_vx = np.array(exp_vx)
    if exp_vy is not None:
        exp_vy = np.array(exp_vy)

    # Start data from 3s (inclusive) and re-base time to 0
    mask = exp_t >=0.01
    exp_t = exp_t[mask]
    exp_x = exp_x[mask]
    exp_y = exp_y[mask]
    if exp_vx is not None:
        exp_vx = exp_vx[mask]
    if exp_vy is not None:
        exp_vy = exp_vy[mask]
    if len(exp_t):
        exp_t = exp_t - exp_t[0]

    # Offset corrections
    # exp_x = exp_x - exp_x[-1]
    # exp_y = exp_y - exp_y[-1]


    # Gaussian smoothing on experimental trajectories to reduce sensor noise
    smooth_sigma = 5  # adjust for stronger/weaker smoothing
    exp_x = gaussian_filter1d(exp_x, sigma=smooth_sigma, mode="nearest")
    exp_y = gaussian_filter1d(exp_y, sigma=smooth_sigma, mode="nearest")

    v0x, v0y = initial_velocity_from_data(exp_t, exp_x, exp_y, exp_vx, exp_vy)

    # print(v0x,v0y)


    sim_T, sim_Pos = e.simulation(exp_x[0]*0.9, exp_y[0]*0.9, v0=(v0x*1.15, v0y*1.15, 0.0))
    sim_T = np.array(sim_T)
    sim_Pos = np.array(sim_Pos)
    sim_x = sim_Pos[:, 0]
    sim_y = sim_Pos[:, 1]
    # print((sim_x[1]-sim_x[0])/0.01, (sim_y[1]-sim_y[0])/0.01)
    # render_trajectory(sim_T, sim_Pos, size=0.2, magnets=[e.magnets[0]], speed=1)

    exp_pos = list()
    for i, j in zip(exp_x, exp_y):
        exp_pos.append([i, j, 0])
    # render_trajectory(exp_t, exp_pos, size = 0.2, magnets = [e.magnets[0]], speed=1)

    # render_d_t(sim_T, sim_Pos, 0)

    exp_t = np.array(exp_t)
    exp_x = np.array(exp_x)
    exp_y = np.array(exp_y)

    sim_amp = amplitude(sim_x, sim_y)
    exp_amp = amplitude(exp_x, exp_y)
    sim_amp_s = gaussian_filter1d(sim_amp, sigma=2, mode="nearest")
    exp_amp_s = gaussian_filter1d(exp_amp, sigma=2, mode="nearest")

    # ----------- Frequency extraction (from amplitude) -----------
    time_amp_s, freq_amp_s, peak_time_amp, freq_amp_raw = freq_from_peaks(sim_T, sim_amp)
    exp_time_amp_s, exp_freq_amp_s, exp_peak_time_amp, exp_freq_amp_raw = freq_from_peaks(exp_t, exp_amp, distance=5)

    # # Try a simple exponential fit on the simulated x-frequency decay
    # if len(time_x_s) > 3.5:
    #     def freq_model(t, f_inf, A, tau):
    #         return f_inf - A * np.exp(-t / tau)

    #     f_inf_guess = np.mean(freq_x_s[-5:]) if len(freq_x_s) > 5 else freq_x_s[-1]
    #     A_guess = f_inf_guess - freq_x_s[0]
    #     tau_guess = (time_x_s[-1] - time_x_s[0]) / 3
    #     popt, _ = curve_fit(freq_model, time_x_s, freq_x_s, p0=[f_inf_guess, A_guess, tau_guess], maxfev=10000)
    #     t_fit = np.linspace(time_x_s[0], time_x_s[-1], 500)
    #     f_fit = freq_model(t_fit, *popt)
    # else:
    #     t_fit, f_fit = np.array([]), np.array([])

    # ----------- Frequency vs time and amplitude vs time -----------
    fig_freq, ax_freq = plt.subplots(figsize=(10, 4))
    if len(freq_amp_s):
        ax_freq.scatter(time_amp_s, freq_amp_s, color="tab:blue", label="Simulated frequency")
    if len(exp_freq_amp_s):
        ax_freq.scatter(exp_time_amp_s, exp_freq_amp_s, color="tab:orange", label="Experimental frequency")
    ax_freq.set_xlabel("Time (s)")
    ax_freq.set_ylabel("Frequency (Hz)")
    ax_freq.set_title("Frequency vs time")
    ax_freq.set_xlim(left=0)
    ax_freq.set_ylim(bottom=0)
    ax_freq.legend()

    fig_amp, ax_amp = plt.subplots(figsize=(10, 4))
    sim_amp_peaks, _ = find_peaks(sim_amp, distance=10)
    exp_amp_peaks, _ = find_peaks(exp_amp, distance=12)
    if len(sim_amp_peaks):
        ax_amp.scatter(sim_T[sim_amp_peaks], sim_amp[sim_amp_peaks], s=20, label="Simulated amplitude", color="tab:blue", alpha=0.8)
    if len(exp_amp_peaks):
        ax_amp.scatter(exp_t[exp_amp_peaks], exp_amp[exp_amp_peaks], s=20, label="Experimental amplitude", color="tab:orange", alpha=0.8)
    ax_amp.set_xlabel("Time (s)")
    ax_amp.set_ylabel("Peak amplitude(m)")
    ax_amp.set_title("Peak amplitude vs time")
    ax_amp.legend()
    
    # Essay graph up
    
    # sim_amp_peaks, _ = find_peaks(sim_amp, distance=10)
    # exp_amp_peaks, _ = find_peaks(exp_amp, distance=12)
    
    # fig_freq, ax_freq = plt.subplots(figsize=(10, 6))
    # ax_amp = ax_freq.twinx()
    # handles = []
    
    

    # if len(freq_amp_s):
    #     h = ax_freq.scatter(time_amp_s, freq_amp_s, color="tab:blue", label="Simulated frequency", s=25)
    #     handles.append(h)
    # if len(exp_freq_amp_s):
    #     h = ax_freq.scatter(exp_time_amp_s, exp_freq_amp_s, color="tab:orange", label="Experimental frequency", s=25, marker="x")
    #     handles.append(h)

    
    # if len(sim_amp_peaks):
    #     h = ax_amp.scatter(sim_T[sim_amp_peaks], sim_amp[sim_amp_peaks], s=18, label="Simulated amplitude", color="tab:green", alpha=0.8, marker="^")
    #     handles.append(h)
    # if len(exp_amp_peaks):
    #     h = ax_amp.scatter(exp_t[exp_amp_peaks], exp_amp[exp_amp_peaks], s=18, label="Experimental amplitude", color="tab:red", alpha=0.8, marker="s")
    #     handles.append(h)

    # ax_freq.set_xlabel("Time (s)")
    # ax_freq.set_ylabel("Frequency (Hz)", color="tab:blue")
    # ax_amp.set_ylabel("Peak amplitude (m)", color="tab:green")
    # ax_freq.set_title("Frequency and amplitude over time")
    # ax_freq.set_xlim(left=0)
    # ax_freq.set_ylim(bottom=0)
    # ax_amp.set_ylim(bottom=0)
    # # ax_freq.grid(True, which="both", axis="both", alpha=0.3)

    # labels = [h.get_label() for h in handles]
    # fig_freq.legend(handles, labels, loc="upper right")
    
    
    
    

    # fig_amp_time, ax_amp_time = plt.subplots(figsize=(16, 7))
    # ax_amp_time.plot(sim_T, sim_amp_s, color="tab:blue", linestyle="-", label="Simulated amplitude")
    # # ax_amp_time.plot(exp_t, exp_amp_s, color="tab:orange", linestyle="-", label="Expermental amplitude")
    # if len(sim_amp_peaks):
    #     ax_amp_time.scatter(sim_T[sim_amp_peaks], sim_amp_s[sim_amp_peaks], color="tab:blue", s=25, marker="o", edgecolor="k", label="Sim peaks")
    # # if len(exp_amp_peaks):
    # #     ax_amp_time.scatter(exp_t[exp_amp_peaks], exp_amp_s[exp_amp_peaks], color="tab:orange", s=25, marker="o", edgecolor="k", label="Experiment peaks")
    #     for idx in range(len(sim_amp_peaks) - 1):
    #         p0, p1 = sim_amp_peaks[idx], sim_amp_peaks[idx + 1]
    #         t0, t1 = sim_T[p0], sim_T[p1]
    #         y_ref = sim_amp_s[p0]
    #         delta_t = t1 - t0
    #         v_offset = 16 + (idx % 3) * 6  # stagger labels a bit to reduce overlap
    #         arrow = FancyArrowPatch((t0, y_ref), (t1, y_ref), arrowstyle="<->", color="tab:blue", lw=1.0, mutation_scale=8, alpha=0.7)
    #         ax_amp_time.add_patch(arrow)
    #         ax_amp_time.annotate(
    #             f"{delta_t:.2f} s",
    #             xy=((t0 + t1) / 2, y_ref),
    #             xytext=(0, v_offset),
    #             textcoords="offset points",
    #             ha="center",
    #             color="tab:blue",
    #         )
    # ax_amp_time.set_xlabel("Time (s)")
    # ax_amp_time.set_ylabel("Amplitude(m)")
    # ax_amp_time.set_title("Amplitude vs time")
    # ax_amp_time.legend()
    # ax_amp_time.set_xlim(left = 0, right = 6.8)
    # ax_amp_time.set_ylim(top=0.12)

    # ----------- Sweep parameters: max frequency vs ma, l, m -----------
    # base_ma = 0.557
    # base_l = 0.292
    # base_m = 0.003976

    # ma_values = np.linspace(0.3, 0.9, 100)
    # l_values = np.linspace(0.24, 0.36, 100)
    # m_values = np.linspace(0.0025, 0.0060, 100)

    # ma_max = np.array([max_freq_for_run(ma=v, l=base_l, m=base_m) for v in ma_values])
    # l_max = np.array([max_freq_for_run(ma=base_ma, l=v, m=base_m) for v in l_values])
    # m_max = np.array([max_freq_for_run(ma=base_ma, l=base_l, m=v) for v in m_values])

    # fig_param, axes_param = plt.subplots(1, 3, figsize=(15, 4))

    # axes_param[0].plot(ma_values, ma_max, marker="o", color="tab:blue")
    # axes_param[0].set_xlabel("Magnetic moment (A·m²)")
    # axes_param[0].set_ylabel("Max frequency (Hz)")
    # axes_param[0].set_title("Max freq vs magnetic moment")
    # axes_param[0].grid(True, alpha=0.3)

    # axes_param[1].plot(l_values, l_max, marker="o", color="tab:orange")
    # axes_param[1].set_xlabel("Pendulum length (m)")
    # axes_param[1].set_ylabel("Max frequency (Hz)")
    # axes_param[1].set_title("Max freq vs pendulum length")
    # axes_param[1].grid(True, alpha=0.3)

    # axes_param[2].plot(m_values, m_max, marker="o", color="tab:green")
    # axes_param[2].set_xlabel("Mass (kg)")
    # axes_param[2].set_ylabel("Max frequency (Hz)")
    # axes_param[2].set_title("Max freq vs mass")
    # axes_param[2].grid(True, alpha=0.3)

    # fig_param.tight_layout()

    # ----------- X/Y vs time overlay -----------
    # fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # ax3.plot(exp_t, exp_x, label="Experimental x(t)", color="tab:orange")
    # ax3.plot(sim_T, sim_x, label="Simulated x(t)", color="tab:blue", alpha=0.8)
    # ax3.set_ylabel("X (m)")
    # ax3.set_title("Trajectory components vs time")
    # ax3.grid(True)
    # ax3.legend()

    # ax4.plot(exp_t, exp_y, label="Experimental y(t)", color="tab:orange")
    # ax4.plot(sim_T, sim_y, label="Simulated y(t)", color="tab:blue", alpha=0.8)
    # ax4.set_xlabel("Time (s)")
    # ax4.set_ylabel("Y (m)")
    # ax4.grid(True)
    # ax4.legend()

    # fig2.tight_layout()
    plt.show()
