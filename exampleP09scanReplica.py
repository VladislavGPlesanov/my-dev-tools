import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# --- Detector geometry ---
n_rows, n_cols = 2, 4
asic_size = 128
height, width = n_rows * asic_size, n_cols * asic_size

# --- Create base array ---
noise_level = 5.61
image = np.ones((height, width)) * 20  + np.random.normal(0, noise_level, (height,width))# uniform background

# --- Define coordinate grid ---
y, x = np.mgrid[0:height, 0:width]

# --- Reverse vertical order of ASICs (so row=0 is bottom row) ---
# Flip Y coordinates to make row 0 bottom
y_flipped = height - y - 1

## --- (1) Gaussian spot in ASIC (0, 3), lower-right corner ---
#asic_r, asic_c = 0, 2
#center_x = asic_c * asic_size + asic_size * 0.25
#center_y = asic_r * asic_size + asic_size * 0.75
#sigma = 5  # narrower than before (previously 10)
#
#gaussian_spot = 350 * np.exp(-(((x - center_x)**2 + (y_flipped - center_y)**2) / (2 * sigma**2)))
#image += gaussian_spot

# --- (1) Asymmetric Gaussian spot in ASIC (0,2), lower-right corner ---
asic_r, asic_c = 0, 2
center_x = asic_c * asic_size + asic_size * 0.25
center_y = asic_r * asic_size + asic_size * 0.75

A = 200
sigma_x = 5.11
sigma_y = 7.35
alpha = 2.73  # skewness parameter: >0 = tail to the right

# Define asymmetric (skewed) Gaussian
dx = x - center_x
dy = y_flipped - center_y
asym_gauss = A * np.exp(-(dx**2 / (2*sigma_x**2) + dy**2 / (2*sigma_y**2))) * (1 + erf(alpha * dx / (np.sqrt(2)*sigma_x)))

# Ensure positivity
asym_gauss = np.clip(asym_gauss, 0, None)
image += asym_gauss

# --- (2) Two vertical Gaussian lines spanning ASICs (0,3) and (1,3) ---
asic_c = 3
start_x = asic_c * asic_size
sigma_line = 5
amplitude_line = 250
curve_strength = 1000

# Two vertical lines, both inside column 3 ASIC region
line1_x = start_x + 40
line2_x = start_x + 80

# curve parabolic outward
offset_X = curve_strength * ((y_flipped - height/2))/ (height/2)**2

for lx in [line1_x, line2_x]:
    curved_x = lx + offset_X
    line_profile = amplitude_line * np.exp(-((x - lx)**2) / (2 * sigma_line**2))
    image += line_profile  # spans both ASIC rows

# --- Plot ---
plt.figure(figsize=(10, 5))
#plt.imshow(image, origin='lower', cmap='inferno', aspect='auto')
plt.imshow(image, origin='lower', cmap='jet', aspect='auto')
plt.colorbar(label='Intensity (a.u.)')
plt.title('Visualisation of the Analyzer X-ray Scan (2×4 ASICs)')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')

# --- Draw ASIC boundaries ---
for c in range(1, n_cols):
    plt.axvline(c * asic_size, color='cyan', lw=0.5, ls='--')
for r in range(1, n_rows):
    plt.axhline(r * asic_size, color='cyan', lw=0.5, ls='--')

plt.tight_layout()
plt.savefig("Example-of-P09-Analyzer.png", dpi=400)
