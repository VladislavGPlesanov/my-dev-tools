import sys
import numpy as np
import re
from time import sleep

def FindTimes(line, keywords):

    pat = fr'(?<={re.escape(keywords)}).*?([-+]?\d*\.\d+|\d+(?:\.\d+)?)(?=\D|$)'
    number = re.findall(pat,line)
    return number


file = open(sys.argv[1],'r')

setup_slacks = []
hold_slacks = []

#re.compile(?<=setup path).*?([-+]?\d*\.\d+|\d+(?:\.\d+)?)(?=\D|$)

#floats = re.compile(r'[-+]?(?:\d*.*\d+)')

n=0
l=0
for line in file:
    this_line = line.split()
    n_words = len(this_line)
    #print(n_words)
    if("Slack" in this_line):
        print(line)
    #elif("Slow" and this_line[5] == "Process" and this_line[6] == "Corner"):
    elif("Slow" in this_line):
        print(line)
    else:
        continue


    




