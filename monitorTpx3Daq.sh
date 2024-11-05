#!/bin/bash

# lets monitor tpx3 data socket 24
#huita=$(ps aux | grep [t]px3_cli |wc -c)
#huita=$(pgrep -c tpx3_cli)
OFILE=ss-output-$(date +"%Y-%b-%m-%H-%M-%S").txt

#echo $huita

while :
do
    if ! pgrep tpx3_cli > /dev/null;
    #if ! pgrep tpx3_gui > /dev/null;
    then
        break
    else
        #echo "tpx3_cli running"
        ss -it | grep "192.168.10.16:24"
        #cat /proc/net/sockstat        
        tstamp=$(date +%s)
        socketinfo=$(ss -it | grep "192.168.10.16:24")
        printf "${tstamp} ${socketinfo}\n" >> $OFILE
        sleep 0.5
    fi
done
