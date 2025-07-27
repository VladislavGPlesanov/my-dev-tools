import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import erf
from scipy.stats import skewnorm


def gauss(x,a,b,c):

    return a * np.exp(-((x-b)**2/(2*c**2)))

def asymFunc(x, alpha, location, scale):

    return 0.5 * (1 + erf(alpha*(x - location)/scale))

def skewedPDF(x, alpha, scale, loc, A, mu, sigma):

    return gauss(x, A, mu, sigma) * asymFunc(x, alpha, loc, scale)

def skewedGauss(x,a,b,c,alpha,pos,omega):

    arg = (x - pos)/omega
    phi = a * np.exp(-((x-b)**2/(2*c)**2))
    PHI = 0.5 * (1+erf(alpha * arg/np.sqrt(2))) 

    return  2.0 / omega * phi * PHI


nbins= 60
data = []

mu = 12
sigma = 1.3

for i in range(1000):
    data.append(sigma*np.random.randn()+mu)
#print(data)

counts, edges = np.histogram(data, bins=nbins)
bin_centers = (edges[:-1]+edges[1:])/2

#plt.figure()
#plt.hist(bin_centers, weights=counts, bins=nbins)
#plt.show()
#plt.close()

xvals = np.arange(0,20,0.1)
yvals = 0.5* (1 + erf((xvals - 4)/1))

y2 = []
y3, y4 = [], []
for i in xvals:
    y2.append(asymFunc(i, 2, 12, 2))
    y3.append(skewedPDF(i, 80, mu, sigma, 2, 12, 2))
    y4.append(skewedGauss(i, 80, 12, 2, -100, 9, 30))

skewness = -3
location = 12
scale = 0.4

pdf = skewnorm.pdf(xvals, skewness, location, scale)

plt.figure()
plt.plot(xvals,yvals, label='yvals')
plt.plot(xvals,y2, label='AsymFunc')
plt.plot(xvals,y3, label='skewedPDF')
plt.plot(xvals,pdf, label='skewnorm')
plt.plot(xvals,y4, label='skewGauss')
plt.legend()
plt.savefig("testplot-asymGauss.png")
plt.close()









