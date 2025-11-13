from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def exponent(x,a,b,c):
    
    return a*np.exp(b*x) + c

def gauss(x,amp,mu,sig):

    return A*np.exp(-(x-mu)**2/(2*sig**2))

def combined_func(x,NA, Asig,musig,sigsig, Abgr, mubgr, sigbgr):

    return (Asig*np.exp(-(x-musig)**2/(2*sigsig**2))*NA) + (Abgr*np.exp(-(x-mubgr)**2/(2*sigbgr**2))*(1-NA))

#def polya():
#
#    return 

x = np.linspace(-2,2,1000)

# gauss pars
A = 5000
mu = 0.6
sig=0.25

# 2nd gauss pars
A2 = 3000
mu2 = 0.15
sig2=0.1

# exponent pars
exp_a = 20
exp_b = 3
exp_c = -200

# convolution

data_exp = exponent(x,exp_a,exp_b,exp_c)
data_gau = gauss(x,A,mu,sig)
data_gau2 = gauss(x,A2,mu2,sig2)
NA = sum(data_gau)/sum(data_gau2)
data_composite = combined_func(x,A,mu,sig,NA,A2,mu2,sig2)

#conv = signal.convolve(data_exp, data_gau, mode='same')
#conv = signal.convolve(data_gau,data_exp,mode='valid')
#conv = signal.convolve(data_gau,data_gau2, mode='same')

plt.figure()
plt.plot(x,gauss(x,A,mu,sig),'b',label="gauss")
plt.plot(x,gauss(x,A2,mu2,sig2),'r',label="gauss")
#plt.plot(x,exponent(x,exp_a,exp_b,exp_c), 'r',label="exponent")
#plt.plot(x,conv,'g',label="convolution")
#plt.plot(x,data_composite,'g',label="composite")
plt.legend()
plt.grid(True)
plt.savefig("CONVO.png")
