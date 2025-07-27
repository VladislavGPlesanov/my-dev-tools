import numpy as np
import matplotlib.pyplot as plt
import sys


lastRun = open('~/IonTestBeams/scintiRun/200umHole.csv','r')

header = None
beamCurrent = None
xzero = None

current, position, rate = [], [], []
e_rate = []
const_e = []

cnt = 0

for line in lastRun:

    if(beamCurrent is None):
        beamCurrent = float(words[0])
    
    xpos = float(words[1])
    itime = words[2].split('/')
    cts = float(words[3])
    factor = float(words[4])
    
    if(xzero is None):
        xzero = xpos
    
    rel_x = xpos - xzero
    
    mean_time = None
    sigma_time = 0.01
    if(len(itime)>1):
       tmp = []
       for i in itime:
            if(len(i)>0):
                tmp.append(float(i))
       mean_time = np.mean(tmp)
       sigma_time = np.sqrt(3*(0.01**2))
    else:
       mean_time = float(itime[0])
    
    irate = (cts*factor)/mean_time
    
    position.append(xpos)            
    #position.append(rel_x)            
    #rate.append((cts*factor)/mean_time)
    const_e.append(sigma_time)
    e_rate.append(getRateErr(irate, cts, mean_time, sigma_time)) 
    rate.append(irate)

    cnt+=1


newRun = open('~/IonTestBeams/scraperRun/beam_profile.csv','r')
