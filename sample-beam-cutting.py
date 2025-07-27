import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Given parameters
beam_flux = 6e9  # particles per second
fwhm_x = 3.0  # mm
fwhm_y = 4.0  # mm
sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
hole_diameter = 0.5  # mm
hole_radius = hole_diameter / 2

# Define x scan range (start 3σ away)
x_start = -3 * sigma_x
x_end = 3 * sigma_x
x_step = 0.5  # mm
x_positions = np.arange(x_start, x_end + x_step, x_step)

# 2D Gaussian function
#def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y):
#    return multivariate_normal.pdf([x, y], mean=[mu_x, mu_y], cov=[[sigma_x**2, 0], [0, sigma_y**2]])

def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y):
    pos = np.column_stack((x, y))  # Ensure correct shape: (N,2)
    return multivariate_normal(mean=[mu_x, mu_y], cov=[[sigma_x**2, 0], [0, sigma_y**2]]).pdf(pos)

# Integrate Gaussian over the hole area
def fraction_through_hole(mu_x, mu_y, sigma_x, sigma_y, hole_radius, num_samples=10000):
    # Generate random points in a circular region
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    r = np.sqrt(np.random.uniform(0, hole_radius**2, num_samples))
    x_samples = mu_x + r * np.cos(theta)
    y_samples = mu_y + r * np.sin(theta)
    
    # Evaluate Gaussian at sampled points
    values = gaussian_2d(x_samples, y_samples, mu_x, mu_y, sigma_x, sigma_y)
    
    # Average the Gaussian values over the hole area
    return np.mean(values)

# Simulate beam movement and compute transmission rate
rates = []
for x_pos in x_positions:
    fraction = fraction_through_hole(x_pos, 0, sigma_x, sigma_y, hole_radius)
    transmitted_rate = fraction * beam_flux
    rates.append(transmitted_rate)
    print(f"x = {x_pos:.2f} mm, Fraction = {fraction:.6f}, Rate = {transmitted_rate:.2e} particles/sec")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x_positions, rates, 'rx-', label="Transmitted Particle Rate")
plt.axvline(0, linestyle="--", color="gray", label="Hole Center")
plt.xlabel("Beam X Position (mm)")
plt.ylabel("Transmitted Rate (particles/sec)")
plt.title("Particle Transmission through Circular Diaphragm")
plt.legend()
plt.grid(True)
plt.savefig("proton_flux.png")


