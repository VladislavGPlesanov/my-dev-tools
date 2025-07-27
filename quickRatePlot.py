import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#def expo(x,a,b,c):
#    return a+b*np.exp(-x*c)

def powerlaw(x,a,b):
    return a*x**(b)


time = [35,18,12,9,7,6,5,5,4,4]
rate = np.linspace(100,1000,10)

guess = [2937,-0.965]

popt,pcov = curve_fit(powerlaw, rate, time, guess)

fitline = powerlaw(np.array(rate), *popt)

plt.figure(figsize=(8,8))
plt.scatter(rate,time,marker="*", c='darkgreen')
plt.plot(rate,fitline,c='darkred')
plt.xlabel(r'$\gamma$ rate, [Hz]')
plt.ylabel('Estimated Measurement time,[min]')
plt.title(r"Measurement time vs $\gamma$ rate")
start = 32
plt.text(600,start,"MDP = 1%,",fontsize=13)
plt.text(600,start-2,r"$\mathrm{ArCO_{2}}$ (80:20)",fontsize=13)
plt.text(600,start-4,r"$\mathrm{L_{Drift}}$ = 2 [cm]",fontsize=13)
plt.text(600,start-6,r"$\mathrm{P}$ = 1 [atm], ",fontsize=13)
plt.text(600,start-8,r"$\epsilon_{Det}\approx22\%$",fontsize=13)
plt.grid(which='major', color='grey', linestyle='-', linewidth=0.5)
plt.grid(which='minor', color='grey', linestyle='--', linewidth=0.25)
plt.savefig("MDP_measurement-RateVsTime-ArCO2.png",dpi=200)
