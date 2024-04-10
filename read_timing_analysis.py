import sys
import numpy as np
import re
from time import sleep

def FindTimes(line, keywords):

    pat = fr'(?<={re.escape(keywords)}).*?([-+]?\d*\.\d+|\d+(?:\.\d+)?)(?=\D|$)'
    number = re.findall(pat,line)
    return number

def findSpaces(word):
    nchar = 0
    for char in word:
        if(char==" "):
            nchar+=1
    return nchar

def stripSpaces(word):
    cleanword = ""
    for char in word:
       if(char!=" "):
            cleanword+=char 
    return cleanword

def checkIfhasClocks(line, clklist):
    found = False
    for item in clklist:
        if(item in line):
           found = True
    return found

def checkStr(line,string):
    if(string in line):
        return True
    else:
        return False 


file = open(sys.argv[1],'r')

t_line = "------------------------------------------------------"
dcr = "Derived Constraint Report"

t_start = 0
t_end = 0
cnt_lines = 0
tab_found = False
prev_line = ""

setup_slacks = []
hold_slacks = []
best_times = []
paths = []

#patt = r'\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*([\d.]+)\s*ns\s*\|\s*([\d.]+)\s*ns\s*\|\s*([\d.]+)\s*ns\s*\|'
clocks = ["TS_tpx3_sfp_CLK320_MMCM", 
          "TS_tpx3_sfp_CLKBUS_MMCM",
          "TS_tpx3_sfp_CLK32_MMCM",
          "TS_SYS_CLK200",
          "TS_CLK125",
          "TS_tpx3_sfp_CLK40_MMCM",
          "ts_rxrecclk"
         ]

pos = {"name":"0", "CHECK":"1", "slack":"2", "bestcase":"3"}

#print(type(pos["name"]))

for line in file:
   if(t_line in line):
     continue
   if("PATH" in line and "TIG" in line and tab_found):
      t_end = cnt_lines
      break
   if("Constraint" in line and "Check" in line):
     t_start = cnt_lines
     tab_found = True
   if(tab_found):
      #found_clk = checkIfhasClocks(line, clocks)
      found_setup = checkStr(line,"SETUP")
      found_hold = checkStr(line,"HOLD")
      if("TIG" in line and "PATH" in line):
          break
      #next_line = next(file,"").strip()
      #print("NL="+next_line)
      #print("[{}] {} [{},{}]".format(cnt_lines, line,found_setup,found_hold))
      spline = line.split("|")
      found_ts = "TS_" in spline[0] or "ts_" in spline[0]
      #print(len(spline))
      #if(found_clk and found_setup):
      if(found_ts and found_setup):
          cleanword= spline[int(pos["name"])].split("=",1)[0]
          stripped = stripSpaces(cleanword)
          #paths.append(spline[int(pos["name"])].split("=",1)[0])
          paths.append(stripped)
      if(found_setup):
          setup_slacks.append(stripSpaces(spline[int(pos["slack"])]))
      if(found_hold):
          hold_slacks.append(stripSpaces(spline[int(pos["slack"])]))
      
   prev_line = line
   cnt_lines+=1  

file.close()

samesize = len(setup_slacks) == len(hold_slacks) and len(setup_slacks) == len(paths)

if(samesize):
    n = 0
    for p in paths:
        print("TIMESPEC:[{}]\t->\tSETUP:{}, HOLD:{}".format(p,setup_slacks[n],hold_slacks[n]))
        n+=1

else:
    print("SETUPS={},HOLDS={},PATHS={}\n".format(len(setup_slacks),len(hold_slacks),len(paths)))
    for item in paths:
       print("{}".format(item))
    for item in setup_slacks:
       print("{}".format(item))
    for item in hold_slacks:
       print("{}".format(item))
    

   













