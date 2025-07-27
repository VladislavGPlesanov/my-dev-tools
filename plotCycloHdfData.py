import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
import tables as tb

infile = sys.argv[1]

base_group_name = '/Raspi F (currently HSR)'

#group = '/Histogram/SeeHorizontal/centers' 
#group = '/Histogram/SeeHorizontal/hist'
group = '/Motorstage/ScanStage' 

with tb.open_file(infile, 'r') as f: 

    beam_centers = np.array(f.get_node(base_group_name+group))
    
    print(len(beam_centers))
    print(type(beam_centers))
    print(type(beam_centers[0]))
    #for i in beam_centers:
    #    print(i)

    print(beam_centers.shape)

#plt.figure()
#plt.plot(beam_centers)
#plt.show()
