import numpy as np
import sys

target_vgrid = float(sys.argv[1])
Vdiff = float(sys.argv[2])

min_increase = 50.0
min_inc_grid = 10.0

L_diff = 2 # cm

Vanode = Vdiff/100.0 + target_vgrid
Vcath = Vdiff*2+Vanode

print(f"V_grid    = {target_vgrid}")
print(f"V_anode   = {Vanode}")
print(f"V_Cathode = {Vcath}")

ramp_base_grid = 200.0
ramp_base_anode = 200.0
ramp_base_cathode = 300.0

grid_range = target_vgrid - ramp_base_grid
anode_range = Vanode - ramp_base_anode
cathode_range = Vcath - ramp_base_cathode

print(grid_range)
print(anode_range)
print(cathode_range)

mid_grid_range = int(np.floor(grid_range/2.0))
print(mid_grid_range)
grid_V = []
step = np.floor((ramp_base_grid+mid_grid_range)/20.0)
print(step)
grid_low = np.linspace(ramp_base_grid, ramp_base_grid+mid_grid_range, int(step))
for i in grid_low:
    grid_V.append(np.floor(i))

print(grid_V)
print(grid_V[1]-grid_V[0])

