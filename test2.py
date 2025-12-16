import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import ListedColormap

class MagneticPendulum:
    def __init__(self, b=0.1, h=0.2, magnet_positions=None, magnet_polarities=None):
        """
        Initialize the magnetic pendulum system.
        
        Parameters:
        - b: damping coefficient
        - h: height of pendulum bob above the magnets
        - magnet_positions: list of (x,y) positions of base magnets
        - magnet_polarities: list of +1 (attractive) or -1 (repulsive) for each magnet
        """
        self.b = b  # damping coefficient
        self.h = h  # height of pendulum bob above magnets
        
        # Default magnet positions (square configuration)
        if magnet_positions is None:
            self.magnet_positions = np.array([
                [1, 1],    # Magnet 1
                [-1, 1],   # Magnet 2
                [-1, -1],  # Magnet 3
                [1, -1]    # Magnet 4
            ])
        else:
            self.magnet_positions = np.array(magnet_positions)
            
        # Default magnet polarities (all attractive)
        if magnet_polarities is None:
            self.magnet_polarities = np.array([1, 1, 1, 1])
        else:
            self.magnet_polarities = np.array(magnet_polarities)
    
    def magnetic_force(self, x, y):
        """Calculate the total magnetic force on the bob at position (x,y)."""
        total_force = np.zeros(2)
        
        for (mx, my), p in zip(self.magnet_positions, self.magnet_polarities):
            dx = x - mx
            dy = y - my
            r_squared = dx**2 + dy**2 + self.h**2
            r_mag = np.sqrt(r_squared)
            
            # Force follows inverse square law (simplified from paper's 1/r^5)
            force_magnitude = p / (r_mag**3)
            
            # Direction is toward (attractive) or away from (repulsive) magnet
            direction = np.array([dx, dy]) / r_mag if p > 0 else -np.array([dx, dy]) / r_mag
            
            total_force += force_magnitude * direction
            
        return total_force
    
    def equations_of_motion(self, state, t):
        """Differential equations governing the pendulum's motion."""
        x, y, vx, vy = state
        
        # Gravitational restoring force (proportional to displacement)
        F_grav = -np.array([x, y])
        
        # Magnetic force
        F_mag = self.magnetic_force(x, y)
        
        # Damping force (proportional to velocity)
        F_damp = -self.b * np.array([vx, vy])
        
        # Total acceleration
        ax = F_grav[0] + F_mag[0] + F_damp[0]
        ay = F_grav[1] + F_mag[1] + F_damp[1]
        
        return [vx, vy, ax, ay]
    
    def simulate(self, initial_state, t_max=50, dt=0.01):
        """Simulate the pendulum's motion from given initial conditions."""
        t = np.arange(0, t_max, dt)
        solution = odeint(self.equations_of_motion, initial_state, t)
        return t, solution
    
    def find_attractor(self, initial_state, t_max=50, dt=0.01, threshold=0.01):
        """
        Simulate until the pendulum settles near an attractor.
        Returns the index of the nearest magnet (0-3) or -1 if none is close.
        """
        t, solution = self.simulate(initial_state, t_max, dt)
        final_pos = solution[-1, :2]
        
        # Find which magnet is closest to final position
        distances = [np.linalg.norm(final_pos - magnet) for magnet in self.magnet_positions]
        min_dist = min(distances)
        closest_magnet = np.argmin(distances)
        
        return closest_magnet if min_dist < threshold else -1
    
    def plot_trajectory(self, initial_state, t_max=50, dt=0.01):
        """Simulate and plot a single trajectory."""
        t, solution = self.simulate(initial_state, t_max, dt)
        x, y = solution[:, 0], solution[:, 1]
        
        plt.figure(figsize=(10, 8))
        
        # Plot magnets
        for i, (mx, my) in enumerate(self.magnet_positions):
            color = 'red' if self.magnet_polarities[i] > 0 else 'blue'
            plt.scatter(mx, my, s=200, c=color, marker='s', label=f'Magnet {i+1}')
        
        # Plot trajectory
        plt.plot(x, y, 'k-', alpha=0.5)
        plt.scatter(x[0], y[0], c='green', s=100, label='Start')
        plt.scatter(x[-1], y[-1], c='black', s=100, label='End')
        
        plt.title('Pendulum Trajectory')
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    def plot_basins(self, x_range=(-2, 2), y_range=(-2, 2), resolution=100, t_max=50):
        """
        Plot the basins of attraction by testing many initial positions.
        
        Parameters:
        - x_range, y_range: ranges for initial positions
        - resolution: number of points in each direction
        - t_max: maximum simulation time for each point
        """
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        
        # Initialize basin grid (-1 means no attractor found)
        basin_grid = -np.ones((resolution, resolution))
        
        # Test each initial position
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                initial_state = [x, y, 0, 0]  # Start from rest
                basin_grid[j, i] = self.find_attractor(initial_state, t_max=t_max)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Colors for each magnet (and white for no attractor)
        colors = ['black', 'red', 'yellow', 'white', 'lightgray']
        cmap = ListedColormap(colors)
        
        # Plot magnets
        for k, (mx, my) in enumerate(self.magnet_positions):
            color = 'red' if self.magnet_polarities[k] > 0 else 'blue'
            plt.scatter(mx, my, s=200, c=color, marker='s')
        
        # Plot basins
        plt.imshow(basin_grid, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                  origin='lower', cmap=cmap, alpha=0.7)
        
        plt.title('Basins of Attraction')
        plt.xlabel('Initial x position')
        plt.ylabel('Initial y position')
        plt.grid(True)
        plt.show()
        
        return basin_grid

# Example usage
if __name__ == "__main__":
    # Create pendulum with default parameters (all magnets attractive)
    pendulum = MagneticPendulum(b=0.1)
    
    # Simulate a single trajectory
    print("Simulating a single trajectory...")
    pendulum.plot_trajectory([1.5, 1.5, 0, 0])
    
    # Plot basins of attraction (this may take a while)
    print("Calculating basins of attraction...")
    pendulum.plot_basins(resolution=100)
    
    # Create pendulum with one repulsive magnet (like Figure 4 in the paper)
    print("Simulating with one repulsive magnet...")
    pendulum_repulsive = MagneticPendulum(b=0.1, magnet_polarities=[1, 1, -1, 1])
    pendulum_repulsive.plot_basins(resolution=3)