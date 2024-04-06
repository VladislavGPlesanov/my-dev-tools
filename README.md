# CONTENTS
## C++ librarry for tpx3 fifo readout

currently two relevant files:
* myReadoutBuffer.cpp (main bulk of code)
  this one contains C++ code with numpy C-API to run tpx3 readout.
* tpx3constants.h (file for future constants to avoid hardcode)

compile with `./compileBuffer.sh`


## what else does what
What present python functions do:

1. hexReg-to-bits.py
decodes MDIO registers read out from an SFP plug by the means of mii-tool
```bash
mii-tool <intf> -vv
```
need to convert thsi output to a text file, from 

```bash
vlad@hostpc:~/readoutSW/tools$ cat regs-enp2s0-glct-03.txt
Using SIOCGMIIPHY=0x8947
enp2s0: negotiated 1000baseT-HD flow-control, link ok
  registers for MII PHY 1: 
    1000 796d 0020 6180 05e1 cde1 000d 0000
    0000 0300 3c00 0000 0000 0000 0000 3000
    0000 0000 0000 0000 0000 0000 0000 4000
    7677 871f 0000 ffff 2801 0000 0000 0000
  product info: vendor 00:10:18 or 00:08:18, model 24 rev 0
  basic mode:   autonegotiation enabled
  basic status: autonegotiation complete, link ok
  capabilities: 1000baseT-HD 1000baseT-FD 100baseTx-FD 100baseTx-HD 10baseT-FD 10baseT-HD
  advertising:  1000baseT-HD 1000baseT-FD 100baseTx-FD 100baseTx-HD 10baseT-FD 10baseT-HD flow-control
  link partner: 1000baseT-HD 1000baseT-FD 100baseTx-FD 100baseTx-HD 10baseT-FD 10baseT-HD flow-control
```
to

```bash
vlad@hostpc:~/readoutSW/tools$ cat regs-mii-tool-vv-noSFP.txt
1000 
7949 
0020 
6180 
05e1 
0000 
0004 
0000
0000 
0300 
0000 
0000
...
... 
```

2. quick\_n\_dirty\_hdf\_read.py
 Code loops through the HDF files in a quick and simple way.

 Params(input converted to string by default):
    filename: name of the file
    option: name of the branch/leaf
 Returns(for each option):
    mask: 2d hist
    thr: 2d hist
    links: link status of RX links int 0 to 7
    config: local chip config
    genconfig: general run configuration
    dacs: DAC settings
    mdata: meta_data 5 lines with a graph of discard and decode errors per chunk
    rdata: raw_data length
usage:
```bash
python3 quick_n_dirty_hdf_read.py <filename> <option>
``` 
3. memoryMonitor.py
 Monitors memory usage for a single PID
 Works fine

4. dataStats.py
 Evaluates number of uniques words in a dat chunk from tpx3 fifo request.

5. CompareErrors.py
 Plots comparison of discard and decode error rates from two hdf files.

## Unfinished scripts:
* while\_loop.py (dont remember what was that for...)
* read-chipscope-waves.py (should allow to draw wavefroms from chipscope (if trigger data saved in a textfile))
* read\_timing\_analysis.py (shoulf got through ise timing files and gather important information)
* testSetup.py (just a simple testground to try out numpy functions...)


