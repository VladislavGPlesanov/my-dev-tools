import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tables as tb


directory = sys.argv[1]
outfilename = sys.argv[2]

flist = glob.glob(directory+'/*.txt')
flist=sorted(flist)

nfiles = len(flist)

FEC, Board, Chip = None, None, None
xhits, yhits, nhits, TOT = [],[],[],[]

cnt = 0
for file in flist:

    data = []
    with open(file,'r') as f:
        data = f.readlines()
    f.close()

    if(FEC is None):
        FEC = int(data[0].split()[1])
    if(FEC is None):
        Board = int(data[1].split()[1])
    if(FEC is None):
        Chip = int(data[2].split()[1])

    ihits = int(data[2].split(",")[1].split(":")[1])
    nhits.append(ihits)

    pixel_data = data[3:]

    if(ihits==0):
        xhits.append([-9999])    
        yhits.append([-9999])    
        TOT.append([-9999])       

    else:
        tmp_x, tmp_y, tmp_tot = [], [], []
        for hit in pixel_data:
            numbers = hit.split()
            x = int(numbers[0])
            y = int(numbers[1])
            q = int(numbers[2])
            tmp_x.append(x)
            tmp_y.append(y)
            tmp_tot.append(q)
        xhits.append(tmp_x)    
        yhits.append(tmp_y)    
        TOT.append(tmp_tot)    
        tmp_x = None
        tmp_y = None
        tmp_tot = None

    try:
        perc = round(float(cnt)/float(nfiles)*100.0,2)
    except ZeroDivisionError:
        perc = 0.0
    print(f"\r{perc}% done", end="",flush=True)

    cnt+=1

with tb.open_file(outfilename+".h5", 'w') as outfile:

    data_group = outfile.create_group(outfile.root, 'data', 'VLArrays')

    filters = tb.Filters(complevel=5, complib='zlib')

    xdata = outfile.create_vlarray(data_group, 'x', tb.Int32Atom(), 'pixel x', filters=filters)
    ydata = outfile.create_vlarray(data_group, 'y', tb.Int32Atom(), 'pixel y', filters=filters)
    TOTdata = outfile.create_vlarray(data_group, 'TOT', tb.Int32Atom(), 'TOT charge', filters=filters)
    nHitsdata = outfile.create_vlarray(data_group, 'nhits', tb.Int32Atom(), 'sum hits', filters=filters)

    nlists = len(xdata)
    print(f"Writing data to hdf file: {outfilename}.h5")
    nconv = 0
    for x,y,tot,nh in zip(xhits, yhits, TOT, nhits):
        xdata.append(np.asarray(x, dtype=np.int32) if hasattr(x,'__len__') else [x])
        ydata.append(np.asarray(y, dtype=np.int32) if hasattr(y,'__len__') else [y])
        TOTdata.append(np.asarray(tot, dtype=np.int32) if hasattr(tot,'__len__') else [tot])
        nHitsdata.append(np.asarray(nh, dtype=np.int32) if hasattr(nh,'__len__') else [nh])
        try:
            perc = round(float(nconv)/float(nlists)*100.0,2)
        except ZeroDivisionError:
            perc = 0.0
        print(f"\r{perc}% done", end="",flush=True)

        nconv+=1



