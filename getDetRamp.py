import numpy as np
import sys

target_vgrid = float(sys.argv[1])
#target_vande = float(sys.argv[2])
#target_vcathode = float(sys.argv[3])

#min_increase = 50.0
#min_inc_grid = 10.0
#cathode_step = int(np.floor(target_vcathode / min_increase))
#vCat_list = []
#
#for i in range(cathode_step+1):
#    vCat_list.append(i*min_increase)
#
#print(f"Part 1:")
#print(vCat_list)
#
#print("Grid, Anode: Beginning")
#
#grid_ramp_range = np.floor(target_vgrid - 200.0)
#grid_ramp_step = int(grid_ramp_range / 10.0)
#grid_V = []
#V_crit = 200.0
#
#for i in range(grid_ramp_step+1):
#    iV = V_crit+i*min_inc_grid
#    grid_V.append(iV)
#
##print(grid_V)
#
#n_initialV = int(np.floor(len(grid_V)/2))
#grid_V_1 = grid_V[0:int(np.floor(len(grid_V)/2))]
##print(grid_V_1)
##print(grid_V_1[0::2])
#gridV_fast = grid_V_1[0::2]
##print(grid_V[n_initialV:])
#gridV_slow = grid_V[n_initialV:]
#gridV_fast.extend(gridV_slow)
#print(gridV_fast)
#
#Vanode = []
#for i in gridV_fast:

Vdiff = float(sys.argv[2])
min_increase = 50.0
min_inc_grid = 10.0

L_diff = 2 # cm

Vanode = Vdiff/10.0 + target_vgrid
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

