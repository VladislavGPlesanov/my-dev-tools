import numpy as np
import matplotlib.pyplot as plt

# Constants
N_particles = int(6e6)  # Monte Carlo samples (scaled down, adjust as needed)
FWHM_x = 3.23  # mm both taken fropm website/experiment section
FWHM_y = 4.72  # mm 
# FWHM = 2.355*sigma
# sigma = FWHM/2.355
#sigma_x = FWHM_x/2.355  # mm (FWHM_x = 3mm)
sigma_x = FWHM_x/(2 * np.sqrt(2 * np.log(2)))# mm (FWHM_x = 3mm)
#sigma_y = FWHM_y/2.355  # mm (FWHM_y = 4mm)
sigma_y = FWHM_y/(2 * np.sqrt(2 * np.log(2)))# mm (FWHM_y = 4mm)

hole_diameters = [0.2, 0.5, 1.0]  # mm (200µm, 500µm, 1000µm)
step_size = 0.5  # mm step in x
#scan_range = 3 * sigma_x  # Move hole from -3σ to +3σ
#scan_range = 6 * sigma_x  # Move hole from -3σ to +3σ
scan_range = 10  # Move hole from -3σ to +3σ
 
#N_per_second = 6e9 # particles per second
N_per_second = 8.6e9 # particles per second
#N_per_second = 0.86e9 # at 100 pA

# Generate Monte Carlo samples
np.random.seed(42)  # Reproducibility
x_particles = np.random.normal(0, sigma_x, N_particles)
y_particles = np.random.normal(0, sigma_y, N_particles)

# Scan hole across x-axis
x_positions = np.arange(-scan_range, scan_range + step_size, step_size)
x_positions_sigma = x_positions/sigma_x

smallest_hole_x = None
smallest_hole_cts = None

# Plot setup
fig, ax = plt.subplots(figsize=(16, 10))

for hole_diameter in hole_diameters:
    hole_radius = hole_diameter / 2  # Convert diameter to radius
    particle_counts = []

    for x_center in x_positions:
        # Find particles inside hole: (x - x_center)^2 + y^2 < hole_radius^2
        distances = (x_particles - x_center)**2 + y_particles**2
        count_inside = np.sum(distances < hole_radius**2)
        particle_counts.append(count_inside)

    # Convert to expected rate per second (scale up to actual rate)
    particle_counts = np.array(particle_counts) * (N_per_second / N_particles)

    # Plot results
    ax.plot(x_positions, particle_counts, marker='x', linestyle='-', label=f"D={hole_diameter * 1e3:.0f} µm")

    if(hole_diameter == 0.2):
        smallest_hole_x = x_positions
        smallest_hole_cts = particle_counts

beamctsstr = ""
beamxstr = ""

print(smallest_hole_x)

for i in range(len(smallest_hole_x)):
    if(smallest_hole_x[i] >= -7.0 and smallest_hole_x[i] <= 2.0):
        beamctsstr += f"{round(smallest_hole_cts[i])},"
        beamxstr += f"{smallest_hole_x[i]},"

print(beamctsstr)
print(beamxstr)
# Final plot styling
ax.set_xlabel("Hole position (mm)")
ax.set_ylabel("Rate [Hz]")
ax.set_title("Proton flux through 3 diaphragms")
# scaled plot for smallest hole
ax.plot(smallest_hole_x, np.array(smallest_hole_cts)*0.014, marker='v', linestyle='--', label=r'$\Phi_{p}$ scaled by 0.014 (LO)')
# ----------------------
ax.legend(title="Hole Size")
sigma_ticks = np.arange(-3,4,1)
sigma_tick_pos = sigma_ticks * sigma_x

plt.vlines(8.5, 10, 1e8, colors='black', linestyles='dashed')
plt.vlines(-8.5, 10, 1e8, colors='black', linestyles='dashed')
plt.hlines(2e6, -12,10, colors='black', linestyles='dashed', label='counter saturation')
#for i in sigma_tick_pos:
#    plt.vlines(i, 0, 1e4, linestyles='dashed')
plt.text(-12.5, 4e6, "Discriminator")
plt.text(-12.5, 2e6, "saturation")
plt.text(-12.5, 1.1e6, "~2MHz")
plt.yscale('log')
plt.ylim([0,1e9])
plt.grid(True)
ax.grid(which='major', color='grey', linestyle='-', linewidth='0.5')
ax.grid(which='minor', color='grey', linestyle='--', linewidth='0.25')


plt.savefig("proton_beam_3holes_1nA.png")

####################################################################

plt.figure(figsize=(8,8))
plt.scatter(smallest_hole_x[0:12], smallest_hole_cts[0:12]/1000, marker='x', c='r', label=r'Expected $\Phi_{p}$')
plt.xlabel("Posiiton x [mm]")
plt.ylabel("Rate [kHz]")
plt.title("Proton rates at the outer beam regions")
plt.grid(True)
########lets scale shit down###############

relative_proton_LO = 0.014
cts_scaled = np.array(smallest_hole_cts[0:12]/1000)*relative_proton_LO

plt.scatter(smallest_hole_x[0:12], cts_scaled, marker='+', c='b', label=r'$\Phi_{p}$ scaled down by LO factor')

plt.legend()
plt.savefig("200umhole-low-rate-region.png")

#######################################################################

bgr_x = np.array([-10.0,-9.5,-9.0,-8.5,-8.0,-7.5,-7.0,-6.5,-6.0,-5.5,-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
bgr_cts = np.array([14,56,208,768,2444,6468,15185,30996,56283,90798,130601,170328,204624,229684,245625,254217,258162,259721,260123,260139,260133,260191,260165,259712,258163,254227,245549,229760,204486,169975,130172,90445,56149,31080,15102,6456,2422,744,219,56,14])

bgr_cts_5mmslit = np.array([0,0,0,0,1,8,56,247,829,2453,6637,15409,31285,56667,91039,130770,170171,203865,228160,242492,246989,242342,228225,203577,169485,129789,90327,56345,31340,15272,6551,2451,802,242,50,5,0,0,0,0,0])

bgr_cts_1mmslit = np.array([0,0,0,0,0,0,0,0,2,5,53,196,655,2107,5517,12487,24356,40723,58770,73644,79228,73549,59193,41184,24598,12576,5626,2199,728,223,61,12,1,1,0,0,0,0,0,0,0])
bgr_cts_2p5mmslit = np.array([0,0,0,0,0,0,1,2,22,116,402,1347,3951,9878,21501,41784,70537,105373,139968,166005,175026,165958,140607,106163,70971,42326,21998,10066,4007,1423,427,122,30,4,1,0,0,0,0,0,0])

print(len(bgr_cts))
print(len(bgr_cts_1mmslit))
print(len(bgr_cts_2p5mmslit))
print(len(bgr_cts_5mmslit))

print("Sizes of position arrays:")
print(len(smallest_hole_x))
print(len(bgr_x))

plt.figure(figsize=(8,8))
plt.scatter(smallest_hole_x, smallest_hole_cts, marker='x', c='r', label=r'Expected $\Phi_{p}$')
plt.scatter(bgr_x, bgr_cts, marker='+', c='b', label=r'Estimated $\Phi_{BGR}$')
plt.xlabel("Posiiton x [mm]")
plt.ylabel("Rate [Hz]")
plt.title("Proton & BGR rates across scan range -10 to 10 mm")
plt.grid(True)
plt.yscale('log')
plt.savefig("200umhole-proton_N-BGR-reates.png")

#######################################################################

ratio_pBGR, ratio_pBGR_5mm = [], []
ratio_pBGR_1mm, ratio_pBGR_2p5mm = [], []

for Np, Nbgr in zip(smallest_hole_cts, bgr_cts):
    ratio_pBGR.append(float(Np)/float(Nbgr))

for Np, Nbgr in zip(smallest_hole_cts, bgr_cts_5mmslit):
    try:
        ratio_pBGR_5mm.append(float(Np)/float(Nbgr))
    except ZeroDivisionError:
        ratio_pBGR_5mm.append(0.0)

for Np, Nbgr in zip(smallest_hole_cts, bgr_cts_1mmslit):
    try:
        ratio_pBGR_1mm.append(float(Np)/float(Nbgr))
    except ZeroDivisionError:
        ratio_pBGR_1mm.append(0.0)

for Np, Nbgr in zip(smallest_hole_cts, bgr_cts_2p5mmslit):
    try:
        ratio_pBGR_2p5mm.append(float(Np)/float(Nbgr))
    except ZeroDivisionError:
        ratio_pBGR_2p5mm.append(0.0)

plt.figure(figsize=(8,8))
plt.scatter(bgr_x,  ratio_pBGR_1mm,   marker='o', c='maroon',  label=r'$L_{slit}$=1mm')
plt.scatter(bgr_x,  ratio_pBGR_2p5mm, marker='o', c='red',     label=r'$L_{slit}$=2.5mm')
plt.scatter(bgr_x,  ratio_pBGR_5mm,   marker='o', c='tomato',  label=r'$L_{slit}$=5mm')
plt.scatter(bgr_x,  ratio_pBGR,       marker='o', c='crimson', label=r'$L_{slit}$=1cm')
plt.xlabel("Position x, [mm]")
plt.ylabel(r"Ratio $N_{protons}/N_{bgr,\gamma}$")
plt.title('Proton to BGR ratio with distance x')
plt.yscale('log')
#plt.ylim([0,400])
plt.grid()
plt.legend()
plt.savefig('Expected_SN_ratio_protons_vs_BGR.png')



