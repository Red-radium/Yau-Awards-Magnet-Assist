import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def render_trajectory(T, Pos, magnets=[], speed=1, size=0.5):
    X = [i[0] for i in Pos]
    Y = [i[1] for i in Pos]
    Z = [i[2] for i in Pos]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory of Single Magnet Pendulum")
    ax.grid()
    ax.set_aspect("equal", adjustable="box")
    ax.scatter([i.pos[0] for i in magnets], [i.pos[1] for i in magnets], c="red", s=100, label="Fixed Magnet", zorder=3)
    line, = ax.plot([], [], c="black", label="Trajectory", zorder=2)
    def init():
        line.set_data([], [])
        return line,
    def update(frame):
        line.set_data(X[:frame+1], Y[:frame+1])
        return line,
    ani = FuncAnimation(fig, update, frames=len(T), init_func=init, blit=True, interval=(T[1]-T[0])*1000/speed)
    ax.legend()
    plt.show()

def render_d_t(T, Pos, index, c="red"):
    X = [i[0] for i in Pos]
    Y = [i[1] for i in Pos]
    Z = [i[2] for i in Pos]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{['X','Y','Z'][index]} (m)")
    ax.set_title(f"{['X','Y','Z'][index]} vs Time")
    ax.plot(T,[X,Y,Z][index],c=c)
    plt.show()

def render_potential_well(x, y, magnets):
    X, Y = np.meshgrid(x, y)
    U = X**2 + Y**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, U, cmap="viridis", alpha=0.8)
    ax.scatter([m[0] for m in magnets], [m[1] for m in magnets], [U[len(y)-1,len(x)-1]], c="red", s=100, label="Fixed Magnet", zorder=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("U(x,y)")
    ax.set_title("Potential Surface")
    ax.legend()
    plt.show()

'''x = np.linspace(-1.5, 1.5, 9)
y = np.linspace(-1.5, 1.5, 9)
render_potential_well(x,y,[[0,0]])'''

