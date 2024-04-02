import sys

def hex_to_binary(hex_number):
    try:
        # Convert hexadecimal to binary
        binary_number = bin(int(hex_number, 16))[2:]
        # Ensure the output is 16 bits long
        binary_number = binary_number.zfill(16)
        # Group into 4-bit groups
        grouped_binary = ' '.join([binary_number[i:i+4] for i in range(0, len(binary_number), 4)])
        return grouped_binary
        #return binary_number[-16:]  # Truncate to 16 bits if longer
    except ValueError:
        print("Invalid hexadecimal number entered.")
        return None

def getRegistersFromFile(infile):
    regs = []
    F = open(infile)
    for line in F:
        words = line.split(" ") 
        #print(len(words)
        if len(words)<12:
            continue
        if("product" in words):
            break
        #print(words)
        for w in words:
            if(len(w)>0):
                if('\n' in w):
                    w = w[:-1]
                regs.append(w)
        #print(regs)
    #print(regs)
    return regs


if __name__ == "__main__":

    regnameList = ["Control","Status","PHY-ID","PHY-ID","AN-Advertisement",
                   "AN-Link-Partner-Ability","AN-Expansion","AN-Next-Page-Transmit",
                   "AN-Next-Page-Receive", "Master-Slave-Control","Master-Slave-status",
                   "PSE-control", "PSE-status","MMD-access-ctrl","MMD-Address-Data-status",
                   "Extended-Status","Vendor-Specific"]

########################################################
    standard_dict = {"basex-AN-ON":[["Control","0"],
                                    ["Status","1"],
                                    ["PHY_ID1","2"],
                                    ["PHY_ID2","3"],
                                    ["AN-Advertisement-Reg","4"],
                                    ["AN-Link-Partner-Ability-Reg","5"],
                                    ["AN-Expansion-Reg","6"],
                                    ["AN-Next-Page-Transmit-Reg","7"],
                                    ["AN-Next-Page-Receive-Reg","8"],
                                    ["Extended-Status-Reg","15"],
                                    ["Vendor-Specific-AN-Interupt-Control","16"]],
                     "basex-AN-OFF":[["Control","0"],
                                     ["Status","1"],
                                     ["PHY-ID1","2"],
                                     ["PHY-ID2","3"],
                                     ["Extended-Status","15"]],
                     "sgmii-AN-ON":[["Control","0"],
                                    ["Status","1"],
                                    ["PHY-ID1","2"],
                                    ["PHY-ID2","3"],
                                    ["AN-Advertisement-Reg","4"],
                                    ["AN-Link-Partner-Ability-Reg","5"],
                                    ["AN-Expansion-Reg","6"],
                                    ["AN-Next-Page-Transmit-Reg","7"],
                                    ["AN-Next-Page-Receive-Reg","8"],
                                    ["Extended-Status","15"],
                                    ["Vendor-Specific-AN-Interupt-Control","16"]],
                      "sgmii-AN-OFF":[["Control","0"],
                                      ["Status","1"],
                                      ["PHY-ID1","2"],
                                      ["PHY-ID2","3"],
                                      ["AN-Advertisement-Reg","4"],
                                      ["Extended-Status","15"]]}
##############################################################
# mii management registers 
#---------------------------------------------------
#  RegAddress | RegName                            |
#---------------------------------------------------
#    0        | Control                            |       
#    1        | Status                             |
#    2,3      | Phy ID                             |
#    4        | AN Advertisement                   |
#    5        | AN Link-Partner base-Page Ability  |                         
#    6        | AN Expansion                       |    
#    7        | AN Next Page Transmit              |              
#    8        | AN Next Page Received Next Page    |                      
#    9        | MASTER-SLAVE control reg           |               
#    10       | MASTER-SLAVE status reg            |               
#    11       | PSE Control                        |  Power Sourcing Equipment (PSE) 
#    12       | PSE Status                         |  
#    13       | MMD Access Control reg             |  MDIO Manageable Device (MMD)
#    14       | MMD Address Data reg               |            
#    15       | Extended Status                    |       
#    16       | Vendor-Specific                    |         
#---------------------------------------------------

####       using:
#print(standard_dict["sgmii-AN-OFF"])
#print(standard_dict["sgmii-AN-OFF"][0])
#print(standard_dict["sgmii-AN-OFF"][0][1])
#####      get:
#[['Control', '0'], ['Status', '1'], ['PHY-ID1', '2'], ['PHY-ID2', '3'], ['AN-Advertisement-Reg', '4'], ['Extended-Status', '15']]
#['Control', '0']
#0
#####


    # Check if the correct number of arguments is provided
    #if len(sys.argv) != 2:
    #    print("Usage: python script_name.py hexadecimal_number")
    #    sys.exit(1)
    cred = "\033[1;31m"
    cgreen = "\033[1;32m"
    cblue = "\033[1;34m"
    clgray = "\033[1;37m"
    cend = "\033[0m"

    otype = sys.argv[1]
    hex_input = sys.argv[2]
    hex_output = []

    hex_input = getRegistersFromFile(sys.argv[2])

    print("got MII Regs")

    if(str(otype)=="num" or str(otype)=="file"): 
        if(str(otype)=="num"):
            binary_output = hex_to_binary(hex_input)
            if binary_output is not None:
                 print("BITS:{}".format(binary_output))
        else:
            regcnt = 0
            #f = open(hex_input)
            print("===========================================")
            #for line in f:
            for line in hex_input:
                temp_binary = hex_to_binary(line)
                hex_output.append(temp_binary)
                if(regcnt<len(regnameList)):
                    if(regcnt==1):
                         print("{}[{}] = \t{} \t{}{}".format(cgreen,
                                                            regcnt,
                                                            temp_binary,
                                                            regnameList[regcnt],
                                                            cend))
                    elif(regcnt==4):
                         print("{}[{}] = \t{} \t{}{}".format(cred,
                                                            regcnt, 
                                                            temp_binary,
                                                            regnameList[regcnt],cend))
                    elif(regcnt==5):
                         print("{}[{}] = \t{} \t{}{}".format(cblue,
                                                            regcnt, 
                                                            temp_binary,
                                                            regnameList[regcnt],cend))
                    else:
                        print("[{}] = \t{} \t{}".format(regcnt, temp_binary, regnameList[regcnt])) 
                else:
                    print("[{}] = \t{} \tNext-Page-Reg".format(regcnt, temp_binary)) 
                regcnt += 1
            print("===========================================")
    else:
        print("usage:\npython3 hexReg-to-bits.py num xxxx\n python3 hexReg-to-bits file <filename.txt>")

####################################################

bits_reg0 = ["Reset","Loopback","Speed Selection LSB","AN enable",
             "Power Down","Isolate", "restart AN", "Duplex Mode",
             "Collision test", "Speed selection MSB", "Unidirectional Enable","Reserved"]

cnt0 = 0;
baseStr = "[{}]={} : {}"
topBit = 15
for i in hex_output[0]:
    if(i==" "):
       continue
    if(cnt0<=11):
        print(baseStr.format(topBit - cnt0, i, bits_reg0[cnt0]))
    else:
        print(baseStr.format(topBit - cnt0, i, bits_reg0[len(bits_reg0)-1]))
    cnt0+=1

clearRegList = []
r_cnt = 0
for reg in hex_output:
    tmp_reg = []
    b_cnt = 0
    for bit in reg:
        if(bit != " "):
            tmp_reg.append(bit)
    clearRegList.append(tmp_reg)

#for rg in clearRegList:
#    print(rg)

tx_test_mode = {clearRegList[9][15],clearRegList[9][14]}
MS_manual_config_fault = clearRegList[10][15] 
MS_manual_config_resolved = clearRegList[10][14] 
MS_local_receiver_stat = clearRegList[10][13] 
MS_remote_receiver_stat = clearRegList[10][12]


   







