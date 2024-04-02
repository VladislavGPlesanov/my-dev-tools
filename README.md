# my-dev-tools
c++ dev for tpx readout various and checking scripts
# C librarry for tpx3 fifo readout

currently two relevant files:
    - myReadoutBuffer.cpp (main bulk of code)
    - tpx3constants.h (file for future constants to avoid hardcode)

compile with `./compileBuffer.sh`

# what else does what
What present python functions do:

## hexReg-to-bits.py
decodes MDIO registers read out from an SFP plug by the means of mii-tool
```bash
mii-tool <intf> -vv
```
## quick\_n\_dirty\_hdf\_read.py
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





