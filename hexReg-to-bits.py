import sys
import glob

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

def getLongestWord(wordlist):
    maxlen = 0;
    for w in wordlist:
        if(len(w)>=maxlen):
            maxlen = len(w);
    return maxlen

def addSpaces(wordlist, maxlen):
    newlist = []
    for w in wordlist:
        length = len(w)
        toAdd = maxlen-length
        spaces = ""
        for i in range(0,toAdd):
            spaces+=" "
        newlist.append(w+spaces)
    return newlist

def getFilelist(dirpath, extension):

    return glob.glob(str(dirpath)+"*mii-tool*."+str(extension))

def getRegistersFromFile(infile):
    regs = []
    F = open(infile)
   # nlines = 0;
   # wlen = []
   # for line in F: 
   #     wlen.append(len(line))
   #     nlines+=1
   # avg_len = sum(wlen)/nlines
   # print("found {} lines with avg {} word length".format(nlines, avg_len))
   # 

    shortFile = False
    cnt = 0;
    checkstr = "Using SIOCGMIIPHY"
    for line in F:
        #if(checkstr in line):
        #   print("PIZDA!")
        #else:
        #   print("HUY!")
        if(cnt==0 and checkstr not in line):
            shortFile=True
            print("File is formated")
        words = None
        #print("[{}] -> {} ->{}".format(len(line),line,shortFile))
        if(shortFile):
           words = line.split("\n")
           #print("SPLITING END-OF-LINE =>> {}".format(words))
           words = words[:-1]
           #print("REMOVING LAST=>> {}".format(words))
        else:
           words = line.split(" ")
           #print("SPLITING SPACES =>> {}".format(words))
        if len(words)<12 and not shortFile:
            #print("skip line with len<12")
            cnt+=1
            continue
        if("product" in words):
            print("exiting file on stop word \"product\"")
            break
        #print(words)
        for w in words:
            #print("HEX=".format(w))
            if(len(w)>0):
                #print("appending {} to regs".format(w))
                if('\n' in w):
                    w = w[:-1]
                regs.append(w)
        cnt+=1
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

    otype = sys.argv[1] # format of the input "num" => hex number, "file"=>txt file, "compare" = comparing multiple files 
    #hex_input = sys.argv[2]
    hex_output = []

    hex_input_list = None

    if(str(otype)=="compare"):
        #flist = list(sys.argv[2:])
        flist = getFilelist(sys.argv[2],"txt")
        print(flist)
        print("reading registers from {} files".format(len(flist)))
        hex_input_list = []
        for f in flist:
            splitname = f.split('/')
            hex_input_list.append([str(splitname[len(splitname)-1]), getRegistersFromFile(f)])
        for l in hex_input_list:
            print("file <<{}>> has registers:\n{}".format(l[0],l[1]))

        status = []  
        an_partner_ability = []        
        an_adv = []
        MS_control = []
        MS_status = []
        an_expansion = []

        for file in hex_input_list:
            for reg in range(0,31):
                if(reg == 1): # control regs
                    filename = file[0]
                    hexnum = file[1][reg]
                    binvalues = hex_to_binary(hexnum)
                    status.append([filename,binvalues])
                if(reg == 5): # link partner ability
                    filename = file[0]
                    hexnum = file[1][reg]
                    binvalues = hex_to_binary(hexnum)
                    an_partner_ability.append([filename,binvalues])
                if(reg == 6): # AN extended
                    filename = file[0]
                    hexnum = file[1][reg]
                    binvalues = hex_to_binary(hexnum)
                    an_expansion.append([filename,binvalues])
                if(reg == 4): # link AN andvrtisement
                    filename = file[0]
                    hexnum = file[1][reg]
                    binvalues = hex_to_binary(hexnum)
                    an_adv.append([filename,binvalues])
                if(reg == 9): # master-slave ctrl
                    filename = file[0]
                    hexnum = file[1][reg]
                    binvalues = hex_to_binary(hexnum)
                    MS_control.append([filename,binvalues])
                if(reg == 10): # master-slave stat
                    filename = file[0]
                    hexnum = file[1][reg]
                    binvalues = hex_to_binary(hexnum)
                    MS_status.append([filename,binvalues])

        reglist = [status,an_adv,an_partner_ability,an_expansion, MS_control, MS_status]
        namelist = ["status",
                    "AN advertisement",
                    "AN link partner abilty",
                    "AN extension",
                    "Master-Slave ctrl",
                    "master-Slave stat"]

        maxlen = 0;
        for item in hex_input_list:
            if(len(item[0])>=maxlen):
                maxlen = len(item[0])

        print("longest name: {}".format(maxlen))

        cnt=0
        for l in reglist:
            print("REGISTER [{}]".format(namelist[cnt]))
            for item in l:
                plugname = item[0]
                if(len(plugname)<maxlen):
                    for i in range(0,maxlen - len(plugname)):
                        plugname+=" "

                if("glct" in item[0] or "wSFP" in item[0]):
                    print("{}{}\t\t{}{}".format(cgreen,plugname,item[1],cend))
                else:
                    print("{}\t\t{}".format(plugname,item[1]))
            print("\n")
            cnt+=1
        

        del hex_input_list

    elif(str(otype)=="num" or str(otype)=="file"): 

        print("got MII Regs")
        print("hex_input is {}".format(hex_input))
        hex_input = getRegistersFromFile(sys.argv[2])

        if(str(otype)=="num"):
            binary_output = hex_to_binary(hex_input)
            if binary_output is not None:
                 print("BITS:{}".format(binary_output))
        else:
            regcnt = 0
            print("===========================================")
            for line in hex_input:
                #print("line={}".format(line))
                temp_binary = hex_to_binary(line)
                #print("decoded={}".format(temp_binary))
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
bits_reg5 = ["PHY Link status", 
             "Acknowledge",
             "Reserved",
             "Duplex",
             "Speed",
             "Reserved(Always 0)",
             "Reserved(Always 1)"
            ]

if(str(otype)=="num" or str(otype)=="file"): 
    cnt0 = 0;
    baseStr = "[{}]={} : {}"
    tableStr1 = [];
    tableStr2 = [];
    topBit = 15
    
    #########################################
    #print("Control:")
    for i in hex_output[0]:
        if(i==" "):
           continue
        if(cnt0<=11):
            str1 = baseStr.format(topBit - cnt0, i, bits_reg0[cnt0])
            tableStr1.append(str1)
        else:
            str1 = baseStr.format(topBit - cnt0, i, bits_reg0[len(bits_reg0)-1])
            tableStr1.append(str1)
        cnt0+=1
    #########################################
    cnt0=0
    # 0   1:9     10:11 12 13 14 15
    # |    |        |    |  |  |  |
    # 0 000000000  00    0  0  0  1
    #baseStr = "[{}]={} : {}"
    #print("AN Link-Partner base-Page Ability:")
    for i in hex_output[5]:
        if(i==" "):
           continue
        if(cnt0<4):
            str2 = baseStr.format(topBit - cnt0, i, bits_reg5[cnt0])
            tableStr2.append(str2)
        elif(cnt0==4 or cnt0==5):
            str2 = baseStr.format(topBit - cnt0, i, bits_reg5[4])
            tableStr2.append(str2)
        elif(cnt0>=6 and cnt0<=14):
            str2 = baseStr.format(topBit - cnt0, i, bits_reg5[5])
            tableStr2.append(str2)
        elif(cnt0>=15):
            str2 = baseStr.format(topBit - cnt0, i, bits_reg5[6])
            tableStr2.append(str2)
        else:
            str2 = baseStr.format(topBit - cnt0, i, "dummy")
            tableStr2.append(str2)
        cnt0+=1
    #########################################
    # adding spaces for formating
    
    maxwidth = getLongestWord(tableStr1)
    newTableStr1 = addSpaces(tableStr1, maxwidth)
    #print(maxwidth)
    #print("\n")
    #for h in tableStr1:
    #    print(len(h))
    #print("\n")
    #for h in newTableStr1:
    #    print(len(h))
    ########################################
    cntr = 0
    print("\nCOMBINED")
    for l in newTableStr1:
        if(cntr==0):
            print("CONTROL \t\t\t\t\t AN Link-Partner base-Page Ability\n")
        out = "{} \t\t\t {}".format(l,tableStr2[cntr])
        print(out)
        cntr+=1
    
    
    
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
    #
    tx_test_mode = {clearRegList[9][15],clearRegList[9][14]}
    MS_manual_config_fault = clearRegList[10][15] 
    MS_manual_config_resolved = clearRegList[10][14] 
    MS_local_receiver_stat = clearRegList[10][13] 
    MS_remote_receiver_stat = clearRegList[10][12]


   







