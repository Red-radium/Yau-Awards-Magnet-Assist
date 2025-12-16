from dipole_terry import *

dx = 0.02
Nx = 10 # number in each direction
dy = 0.02
Ny = 10

Sx = -Nx*dx
Sy = -Ny*dy

field = np.zeros((Ny*2,Nx*2))

e = Experiment()

import matplotlib.pyplot as plt
from renderer import *

T,Pos = e.simulation(0.05,0.1)
print(Pos[-1])
print(len(T))
render_trajectory(T,Pos,magnets=e.magnets,speed=1,size=0.2)

'''T,Pos = e.simulation(-0.13,0.1)
print(Pos[-1])
print(len(T))
render_trajectory(T,Pos,magnets=e.magnets,speed=1,size=0.2)'''


for ix in range(-Nx,Nx):
    for iy in range(-Ny,Ny):
        x = ix*dx
        y = iy*dy
        #print(x,y)
        T,Pos = e.simulation(x,y)
        print(len(T))
        field[Ny+iy][Nx+ix] = Pos[-1][0]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
im = ax.imshow(field, cmap="bwr")
ax.set_box_aspect(1)
fig.tight_layout()
plt.show()
