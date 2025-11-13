import numpy as np
from scipy.special import erfi, gamma
import matplotlib.pyplot as plt

def plotFuncs(x,y,labels,funclabels,picname):

    plt.figure(figsize=(16,8))
    cnt=0
    huya = True if "polya2" in picname else False
    for fn in y:
        if(huya):
            #plt.plot(x,fn,label=f"{funclabels[cnt]}")
            plt.scatter(x,fn,label=f"{funclabels[cnt]}")
        else:
            plt.plot(x,fn,label=f"{funclabels[cnt]}")
        cnt+=1
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.ylim([1e-8,1])
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"{picname}.png")

def scurve(x,A,mu,sigma):
    return 0.5 * A * erf((x - mu)/(np.sqrt(2)*sigma)) + 0.5 * A 

def polyaOne(a, x, b, c):

    Qx = a*(x**b)*np.exp(-c*x)

    return Qx

def polyaTwo(n, mean, var):
    #from sigma^2 = mean*(theta+1) express (theta+1), which is var^2/mean^2 !

    try:    
        result = (1/mean)*(np.power((var**2/mean**2),(var**2/mean**2)))/(gamma(var**2/mean**2))*np.power((n/mean),(var**2/(mean**2 - 1)))*np.exp(-(var**2/mean**2)*(n/mean))
    except:
        result = -999

    return result

##################################################################

#x = np.linspace(0,5000)

#gain1 = []
gains2= []

n_ = np.linspace(0,5000)

print(len(n_))

#for point in x:
#    #                     a,x,b,c
#    gain1.append(polyaOne(2,0.5,3,0.45))
#    #                       

#var = np.linspace(100,1000,100)
#var = [100,200,300,400,500,600,700,800,900,1000,1200,1500,2000,5000,10000,200000]
#var = [100,200,300,400,500,600,700,800,900,1000,1200,1500,2000,5000,10000]
var = [10,50,100,200]
mean = [5,10,15,20,25,30]
fuLabels = []

for m in mean:
    for v in var:
        tmp = []
        fuLabels.append(f"mean={m}, var={v}")
        for point in n_:
            tmp.append(polyaTwo(float(point), float(m), float(v)))
        fact = 1/(float(v)**2/m**2)
        print(f"mean={m} and variance={v} yields factor f={fact}")
        gains2.append(tmp)
        tmp=None

#print(gain1[0:10])
#print(gain2[0:10])

#plotFuncs(x, [gain1], ["examplegian","n","P(n)"], ["a*x^b*exp(-cx)"],"gains-polya1")
plotFuncs(n_, gains2, ["examplegian","n","P(n)"], fuLabels ,"gains-polya2")




#plt.plot(x, erf(x))
#        x,        x, A, mu, sigma
#plt.plot(x, scurve(x, 4, mu, sigma),label="sig=0.5")
##plt.plot(x, scurve(x, 3, 1, 0.6),label="sig=0.6")
##plt.plot(x, scurve(x, 2, 1, 0.7),label="sig=0.7")
##plt.plot(x, scurve(x, 1, 1, 1.0),label="sig=1")
#plt.vlines(mu,0,4,colors='cyan',linestyles='-')
#plt.vlines(mu+sigma,0,4,colors='black',linestyles='--')
#plt.vlines(mu-sigma,0,4,colors='black',linestyles='--')
#plt.xlabel('$x$')
#plt.ylabel('$erf(x)$')
#plt.legend(loc="upper left")
#plt.show()
