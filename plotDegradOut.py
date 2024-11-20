import numpy as np
import matplotlib.pyplot as plt
from MyPlotter import myPlotter
from MyPlotter import myUtils 
import sys
import glob
import os
from operator import add

MU = myUtils()

location = sys.argv[1] # path to degrad.OUT files 
prefix = sys.argv[2]
txtFile = sys.argv[3]

dir_files = glob.glob(location+"*.OUT")
file_list = sorted(dir_files)
print(f"found <{len(file_list)}> degrad output files")
nFiles = len(file_list)
for i in range(10):
    print(f"file{i}->{file_list[i]}")

good_events = None
with(open(txtFile, "r") as txt):
    
    string = txt.readline()
    good_events = string.split(',')

#print(type(good_events))
#print(good_events[0:10])

#exit(0)
x_pe = np.array([])
y_pe = np.array([])
z_pe = np.array([])
t_pe = np.array([])
# ------ good events -------
x_pe_good = np.array([])
y_pe_good = np.array([])
z_pe_good = np.array([])
t_pe_good = np.array([])

##frange = np.linspace(minf,maxf,1)

l_empty = []
not_empty = []

n_fluor=0
n_pairProd=0
n_bstrah=0
# ----- header counters -----------
event_char = [ 0 for i in range(9)]
print(event_char)
# -------------------------------

cnt = 0
nEmpty = 0
for f in file_list:

    fname = MU.removePath(str(f))

    evtNr = int(fname[16:21])
    #print(f"this is event {evtNr}")    
     
    if(os.stat(f).st_size==0):
        l_empty.append(cnt)
        cnt+=1
        nEmpty+=1
        continue

    with open(f, "r") as file:
        header = file.readline()  # Read the header (first line)
        long_line = file.readline()  # Read the second long line

    header_split = list(map(float, header.split()))
    this_event = header_split[0]
    event_char = list(map(add, event_char, header_split[3:]))

    #if(cnt==10):
    #exit(0)
    # Split the long line into individual numbers
    numbers = list(map(float, long_line.split()))
    
    # Ensure the number of elements is divisible by 7
    if len(numbers) % 7 != 0:
        raise ValueError("The number of elements in the line is not a multiple of 7!")
    
    # Reshape into rows of 7 columns (each row corresponds to a particle)
    data = np.array(numbers).reshape(-1, 7)
    
    # Extract the x, y, z, t parameters (columns 0 to 3)
    #x, y, z, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    x, y, z, t, f_Fluor, f_pairProd, f_bstrah = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:,4], data[:,5],data[:,6]
   
    #print("x={}, last 4={}".format(len(x), x[-4:])) 
    #print("y={}, last 4={}".format(len(y), y[-4:]))
    #print("z={}, last 4={}".format(len(z), z[-4:]))
    #print("t={}, last 4={}".format(len(t), t[-4:]))

    #print(f"Event {evtNr} from filename")

    if(str(evtNr) in good_events):
        #print(f"Found good one at {evtNr}")
        x_pe_good = np.append(x_pe_good,x)
        y_pe_good = np.append(y_pe_good,y)
        z_pe_good = np.append(z_pe_good,z)
        t_pe_good = np.append(t_pe_good,t)
 
    else:
        #print(f"Found OTHER one at {evtNr}")
        x_pe = np.append(x_pe,x)
        y_pe = np.append(y_pe,y)
        z_pe = np.append(z_pe,z)
        t_pe = np.append(t_pe,t)

    n_fluor+=(np.sum(f_Fluor)) 
    n_pairProd+=(np.sum(f_pairProd)) 
    n_bstrah+=(np.sum(f_bstrah)) 

    x, y, z, t, f_Fluor, f_pairProd, f_bstrah = None, None, None, None, None, None, None 

    not_empty.append(cnt)

    #print(len(x_pe))
    MU.progress(nFiles, cnt)
    
    cnt+=1

print(f"\nFound <{nEmpty}> empty files out of <{cnt}>")
print(f"{l_empty[0:30]}")
print(f"{not_empty[0:30]}")

print(f"Lenght of good event clusters -> {len(x_pe_good)}")

names = ["EXC GAS1", "EXC GAS2", "EXC GAS3", "EXC GAS4","EXC GAS5", "EXC GAS6", "N PRIM COLL COMPTON", "N PRIM PAIRPROD COLL", "NULL COLL CLUST SIZE (MOL.GAS)"]

for i,j in zip(names, event_char):
    print(f"\t{i} = {j}")


print(f"Found Fluorescence events={n_fluor}, pair production={n_pairProd}, bremsstrahlung={n_bstrah}")

mp = myPlotter()

mp.simpleHist(x_pe, 51, np.nanmin(x_pe)*0.9, np.nanmax(x_pe)*1.1, ["Prim-electron-x","x","N"], f"{prefix}-PE-x", ylog=True)
mp.simpleHist(y_pe, 51, np.nanmin(y_pe)*0.9, np.nanmax(y_pe)*1.1, ["Prim-electron-y","y","N"], f"{prefix}-PE-y", ylog=True)
mp.simpleHist(z_pe, 51, np.nanmin(z_pe)*0.9, np.nanmax(z_pe)*1.1, ["Prim-electron-z","z","N"], f"{prefix}-PE-z", ylog=True)
mp.simpleHist(t_pe, 51, np.nanmin(t_pe)*0.9, np.nanmax(t_pe)*1.1, ["Prim-electron-t","t","N"], f"{prefix}-PE-t", ylog=True)

mp.simpleHist(x_pe_good, 51, np.nanmin(x_pe_good)*0.9, np.nanmax(x_pe_good)*1.1, ["Prim-electron-x","x","N"], f"{prefix}_good-PE-x", ylog=True)
mp.simpleHist(y_pe_good, 51, np.nanmin(y_pe_good)*0.9, np.nanmax(y_pe_good)*1.1, ["Prim-electron-y","y","N"], f"{prefix}_good-PE-y", ylog=True)
mp.simpleHist(z_pe_good, 51, np.nanmin(z_pe_good)*0.9, np.nanmax(z_pe_good)*1.1, ["Prim-electron-z","z","N"], f"{prefix}_good-PE-z", ylog=True)
mp.simpleHist(t_pe_good, 51, np.nanmin(t_pe_good)*0.9, np.nanmax(t_pe_good)*1.1, ["Prim-electron-t","t","N"], f"{prefix}_good-PE-t", ylog=True)



