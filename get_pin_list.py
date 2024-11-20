import pandas as pd
import sys

csvfile = sys.argv[1]

if(csvfile=="-h" or csvfile=="--help"):
    print("put file name alike \"ise/tpx3_ml605_pad.csv\"")
    exit(0)
#data = pd.read_csv('ise/tpx3_ml605_pad.csv', skiprows=20)
data = pd.read_csv(csvfile, skiprows=20)

data.columns = ["Pin Number",
                "Signal Name",
                "Pin Usage",
                "Pin Name",
                "Direction",
                "IO Standard",
                "IO Bank Number",
                "Drive (mA)",
                "Slew Rate",
                "Termination",
                "IOB Delay",
                "Voltage","Constraint","IO Register","Signal Integrity","yoba"]

signals = []
cnt = 0
for i in data['Signal Name']:
    cnt += 1
    if(str(i) != "nan"):
       filler = ""
       print("found:{} \t\t pin={}".format(i,data['Pin Number'][cnt-1]))
    


#print(signals)
#print(data[['Signal Name']].to_string(index=False))
