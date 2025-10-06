import sys
import numpy as np

def calcTime(T1, T2):

    #return 256**T1 *46 * T2 / 40e9
    return np.power(256,T1) *46 * T2 / 40e6

SR = int(sys.argv[1])
ST = int(sys.argv[2])

time = calcTime(SR,ST)

print(f"Frame time is : {time*1000.0:.4f} ms")

