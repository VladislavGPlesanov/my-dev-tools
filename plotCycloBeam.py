import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit

def exponent(x, a, b, c):
    return a * np.exp(b*x) + c

def gauss(x, A, mu, sigma):
    return A*np.exp(-((x-mu)**2)/(2*sigma**2))

def linear(x,a,b):
    return a*x + b

def getRateErr(rate, cts, t, t_e):
    # set sigma_current to be 1 pA
    return rate*np.sqrt((np.sqrt(cts)/cts)**2 + (t_e/t)**2)

infile = sys.argv[1]
datatype = int(sys.argv[2])
picname = sys.argv[3]

if(datatype != 1 and datatype != 2 and datatype !=3):
    print(f"Eblo, no such option as {datatype}! only 1,2,3")

header = None
beamCurrent = None
xzero = None

f = open(infile)

current, position, rate = [], [], []
e_rate = []
const_e = []

cnt = 0
for line in f:
    words = line.split(',')
        
    if(cnt==0):
        header = words
    else:
        if(datatype == 1):
            cts = float(words[0])
            itime = float(words[1])
            Ibeam = float(words[2])
            current.append(Ibeam)
            rate.append(cts/itime)
            time_err = 0.01
            e_rate.append(getRateErr(cts/itime, cts, itime, time_err))            
            const_e.append(1) 
            cts, itime, Ibeam = None, None, None

        elif(datatype==2):
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
        else:

            print('PEZDA!')
 
    cnt+=1
    #print(cnt)


plt.figure(figsize=(12,10))
if(datatype==1):
    #plt.scatter(current, rate)
    plt.errorbar(current, rate, xerr = const_e, yerr = e_rate, fmt='x', color='red', ecolor='black', capsize=2)
    plt.title("Beam current vs BGR rate")
    plt.xlabel(header[2])
    plt.ylabel('rate[Hz]')
    plt.grid(True)
    plt.xlim([20,1050])
    plt.ylim([5,220])
    # --- fit line ---  
    popt, pcov = curve_fit(linear, current, rate)
    print(f"fitting line\nslope={popt[0]}, offset={popt[1]}")
    line_func = linear(np.array(current), *popt)
    plt.plot(current, line_func, label='linfit', c='b', linestyle=':')
    plt.text(100, 200, "Shielding:", bbox=dict(facecolor='grey', alpha=0.25, edgecolor='white'))
    plt.text(100, 195, "6 cm Al, 15 cm Pb bricks", bbox=dict(facecolor='grey', alpha=0.25, edgecolor='white'))
    #---------------------------
    plt.savefig("IbeamVsBGR-"+picname+".png")
else:
    print("eblo")
    plt.errorbar(position, rate, xerr = const_e, yerr = e_rate, fmt='x', color='blue', ecolor='black', capsize=2, label=r"$\Phi$ Detected")
    # ---------  
    max_rate = np.max(rate)
    #print(max_rate)
    plt.hlines(max_rate, 10, 27.5, colors='red', linestyles='--')
    plt.text(10, 2e5, f"Max rate achieved = {round(max_rate/1000,2)} kHz",color='red' )
    sorted_rate = sorted(rate)
    max_r1 = sorted_rate[len(sorted_rate)-1]
    max_r2 = sorted_rate[len(sorted_rate)-2]
    index_rate1 = rate.index(max_r1)
    index_rate2 = rate.index(max_r2)
    pos_r1 = position[index_rate1]
    pos_r2 = position[index_rate2]
    
    middle = (pos_r1 - pos_r2)/2
    #print(pos_r2)
    #print(pos_r1)
    #print(pos_r1-pos_r2)
    #print((pos_r1-pos_r2)/2)
    hole_position = pos_r2+middle

    plt.vlines(hole_position, 100, 1e5, colors='green', linestyles='dashed')
    plt.text(pos_r2-3.5, 1e3, f"200um hole @ x={hole_position}", color='green')

    last_bgr_rate = rate[3]
    last_bgr_pos = position[3]
    
    plt.vlines(last_bgr_pos, 100, 500, colors='magenta', linestyles='dashed')
    plt.arrow(last_bgr_pos, 150, (hole_position - last_bgr_pos)-0.25, 0, head_width=10, head_length=0.25, fc='k', ec='k') 
    plt.text(15.5, 155, f"no signal to peak = {hole_position - last_bgr_pos} mm", color='black')

    # lines at which should see shit
    #plt.vlines(hole_position-6, , 500, colors='magenta', linestyles='dashed')
    expected_rate = np.array([0,0,1433,10033,44433,124700,399900,997600,2224533,4117967,6984633,10388800,13545000,15554533,15127400,13663967,10166633,6867100])
    expected_x = np.array([-7,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2])

    visible_bgr_pos = np.array([-10.0,-9.5,-9.0,-8.5,-8.0,-7.5,-7.0,-6.5,-6.0,-5.5,-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0])
    visible_bgr_cts = np.array([17,61,229,739,2364,6471,15045,31121,56365,90460,130119,170232,204343,229663,245477,254162,258213,259669,260076,260129,260149,260173,260062,259742,258229])
      
    visible_bgr_cts_5mmslit = np.array([0,0,0,0,2,10,60,221,793,2473,6505,15113,31005,55808,89875,129576,169480,203542,228286,242567,247308,242741,228559,204457,170628])
  
    #plt.scatter(expected_x+23.5, expected_rate*0.014, marker='+', c='brown', ls='-',  label=r"$\Phi_{p, Expected}$")
    #plt.scatter(visible_bgr_pos+23.5, visible_bgr_cts*0.35, marker='v', c='olive', ls='-', label=r"$\Phi_{bgr, visible}$")
    # --------------------------------

    plt.plot(expected_x+23.5, expected_rate*0.014, marker='+', c='brown', ls='-',  label=r"$\Phi_{p,expected}\cdot LO$ (0.014 for $E_{p}$=13.16 MeV)")
    plt.plot(visible_bgr_pos+23.5, visible_bgr_cts*0.35, marker='v', c='olive', ls='-', label=r"$\Phi_{bgr, visible}$")

    # ---------  
    plt.xlabel(header[1])
    plt.ylabel('rate[Hz]')
    plt.title('Al diaphragm position vs scinti rate')
    plt.yscale('log')
    #plt.ylim([10.0,300000.0])
    plt.grid(which='major', color='grey', linestyle='-',linewidth=0.5)
    plt.grid(which='minor', color='grey', linestyle='--',linewidth=0.25)

    #plt.legend(loc='center left')
    plt.legend(loc='lower right')
    
    plt.savefig("BeamPositionVsRate-"+picname+".png")


