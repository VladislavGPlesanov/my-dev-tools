import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Beam properties
FWHM_x = 3.0  # mm
FWHM_y = 4.0  # mm
sigma_x = FWHM_x / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
sigma_y = FWHM_y / (2 * np.sqrt(2 * np.log(2)))

# Monte Carlo sampling
#num_particles = int(1e6)  # Number of Monte Carlo samples
#num_particles = int(3.019176e6)  # total nr of gamma bgr per second (From Hemanth's data)
num_particles = int(2.60774e5)  # total nr of gamma bgr per second
x_samples = np.random.normal(0, sigma_x, num_particles)
y_samples = np.random.normal(0, sigma_y, num_particles)

# Slit properties
#slit_width = 1.0  # mm
#slit_width = 10.0  # mm
#slit_width = 5.0  # mm
slit_width = [1.0, 2.5, 5.0, 10.0]  # mm
slit_height = 50.0  # mm

# Circular blocker properties (200 µm = 0.2 mm diameter → radius = 0.1 mm)
blocker_radius = 0.2 / 2  

# Define slit positions in mm
x_positions = np.arange(-10, 10.5, 0.5)  # Move from -10 mm to +10 mm in 0.5 mm steps
sigma_positions = x_positions / sigma_x  # Convert to sigma_x units
particles_through_slit = []

x_vals = np.linspace(-10, 10, 1000)
beam_profile = norm.pdf(x_vals, loc=0, scale=sigma_x) * num_particles

fig, ax1 = plt.subplots(figsize=(8, 6))

clrs = ['darkblue','slateblue', 'blueviolet','purple']
mrkrs = ['o','x','+','v']
n_slit = 0
# Move slit across beam for each size of slit
for sl in slit_width:
    particles_through_slit = []
    for x_slit_center in x_positions:
        # Define slit boundaries
        #x_min = x_slit_center - slit_width / 2
        x_min = x_slit_center - sl / 2
        #x_max = x_slit_center + slit_width / 2
        x_max = x_slit_center + sl / 2
        y_min = -slit_height / 2
        y_max = slit_height / 2
    
        # Mask particles that pass through the slit
        mask_slit = (x_samples >= x_min) & (x_samples <= x_max) & (y_samples >= y_min) & (y_samples <= y_max)
    
        # Mask particles that are blocked by the circular blocker at slit center
        mask_blocker = (x_samples - x_slit_center) ** 2 + y_samples ** 2 <= blocker_radius ** 2
    
        # Compute final number of particles that pass **through slit but not blocked**
        mask_final = mask_slit & ~mask_blocker
        particles_through_slit.append(np.sum(mask_final))

    #posstr = ""
    #Ngammastr = ""
    #
    #print(f"Reference numbers for slit size:{sl}\n")
    #for i in range(len(x_positions)):
    #    #if(x_positions[i]>=-10.0 and x_positions[i]<= 2.0):
    #    posstr+=f"{x_positions[i]},"
    #    Ngammastr+=f"{particles_through_slit[i]},"  
    #
    #print(f"Desired positions:\n{posstr}\n")
    #print(f"Desired counts:\n{Ngammastr}\n")

    #print("------------------------------------------------")
        
    #ax1.plot(x_positions, np.array(particles_through_slit)/1000, marker='o', linestyle='-', color=clrs[n_slit], label=r"$r_{\gamma}$, $L_{slit}$ = "+f" {sl} mm")
    ax1.plot(x_positions, np.array(particles_through_slit)/1000, marker=mrkrs[n_slit], linestyle='-', color=clrs[n_slit], label=r"$r_{\gamma}$, $L_{slit}$ = "+f" {sl} mm")
    n_slit+=1

ax1.plot(x_vals, beam_profile/1000, linestyle="dashed", color="orange", label="Secondary Beam Profile")

ax1.set_xlabel(r"Secondary Beam Position relative to 200$\mu$m hole (mm)")
ax1.set_ylabel("Rate @ scintillator, [kHz]")
ax1.set_title("Secondary beam rate within unshielded Al part")
ax1.grid(True)
ax1.legend(loc='upper right')

#plt.vlines(-5,0,350, colors='black', linestyles='dashed')
#plt.vlines(5,0, 350, colors='black', linestyles='dashed')

plt.savefig("secondaryRate_throughSlit.png")





