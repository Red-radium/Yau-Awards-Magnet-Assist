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
    def __init__(self,x=0.0,y=0.0):
        ma = 1.0
        self.line_length = 0.4
        self.hanging_point = np.array([0.0,0.0,0.45])
        self.m = 0.02
        self.pos = np.array([0.0,0.0,0.0])
        self.ma = np.array([0.0,0.0,ma])
        self.pos[0] = x
        self.pos[1] = y
        self.dampingco = 0.01
        if x**2 + y**2 > self.line_length:
            raise Exception(f"line length {self.line_length} is not enough to get ({x},{y})")
        else:
            self.pos[2] = self.hanging_point[2] - np.sqrt(self.line_length**2 - self.pos[0]**2 - self.pos[1]**2)
        #print(self.pos)
        self.v = np.array([0.0,0.0,0.0])
        self.magnets = [Magnet([-0.1,0.0,0.0],[0.0,0.0,ma]),Magnet([0.1,0.0,0.0],[0.0,0.0,ma])]
        # self.magnets = [Magnet([0,0,0],[0,0,0])]
        self.dt = 0.01 #0.01 # dt
    def motion_with_tension(self):
        pos = self.pos
        vel = self.v
        m = self.m
        dmp = self.dampingco

        gravity = np.array([0, 0, -m * g])
        magnetic_force = 0
        for magnet in self.magnets:
            magnetic_force += dipole_force(magnet.ma, self.ma, magnet.pos, self.pos)
        damping = -dmp * vel

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
    def simulation(
        self,
        x=0.0,
        y=0.0,
        max_time=70.0,
        terminate_acc=0.01,
        terminate_vel=0.02,
        respect_termination=True,
        v0=None,
    ):
        # Re-init state with optional custom initial velocity
        self.__init__(x, y)
        if v0 is not None:
            self.v = np.array(v0, dtype=float)
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
            # print(rhat)

            #    so v is purely tangential—no energy kick!
            self.v -= np.dot(self.v, rhat) * rhat

            t += self.dt
            T.append(t)
            Pos.append(self.pos.copy())
            if (
                respect_termination
                and np.linalg.norm(acc) <= terminate_acc
                and np.linalg.norm(self.v) <= terminate_vel
            ):
                #print(acc)
                #print(self.v)
                break
            #print(self.pos)
        return T,Pos

# e = Experiment()
# import matplotlib.pyplot as plt
# T,Pos = e.simulation(0.038,0.039)
# print(Pos[:10])
# from renderer import *
# render_trajectory(T,Pos,size=0.2,magnets=e.magnets,speed=1)
# render_d_t(T,Pos,0)
#plt.plot(T,[i[2] for i in Pos])
#plt.show()
# import matplotlib.pyplot as plt
# from renderer import *


def load_damping_data(path="damping1.txt"):
    """Read damping measurements as time, x, y lists."""
    data = np.genfromtxt(path, skip_header=2, delimiter="\t")
    if data.ndim == 1:
        data = data[np.newaxis, :]  # ensure 2D
    t_vals = data[:, 0].tolist()
    x_vals = data[:, 1].tolist()
    y_vals = data[:, 2].tolist()
    x_vals = [n+0.003 for n in x_vals]
    y_vals = [n+0.008 for n in y_vals]
    
    t_vals = t_vals[:4200]
    x_vals = x_vals[:4200]
    y_vals = y_vals[:4200]
    return t_vals, x_vals, y_vals

# damp_t, damp_x, damp_y = load_damping_data()

# sim_x = [p[0] for p in Pos]
# sim_y = [p[1] for p in Pos]

# fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)

# # axes[0].plot(T, sim_x, label="Simulated trajectory", c="tab:blue", alpha = 0.5)
# # axes[0].plot(damp_t, damp_x, label="Empirical trajectory", c="tab:orange", alpha = 0.5)
# axes[0].plot(damp_t, damp_x, label="Empirical trajectory", c="tab:orange", alpha = 1)
# axes[0].set_xlabel("Time (s)")
# axes[0].set_ylabel("X (m)")
# axes[0].set_title("X vs Time")
# axes[0].legend()

# # axes[1].plot(T, sim_y, label="Simulated trajectory", c="tab:blue", alpha = 0.5)
# # axes[1].plot(damp_t, damp_y, label="Empirical trajectory", c="tab:orange", alpha = 0.5)
# axes[1].plot(damp_t, damp_y, label="Empirical trajectory", c="tab:orange", alpha = 1)
# axes[1].set_xlabel("Time (s)")
# axes[1].set_ylabel("Y (m)")
# axes[1].set_title("Y vs Time")
# axes[1].legend()

# fig.tight_layout()
# plt.show()

