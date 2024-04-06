###################################################################################
#
# Should adjust this one later for programming based on a batch file from input
#
##################################################################################

rm -f batch_temp.cmd

bitfile=$1
board=$2

pos=0
if [[ $board == "ML605" ]]; then
    pos=2
elif [[ $board == "FECv6" ]]; then 
    pos=1
else
    pos=0        
fi

if [[ -z $XILINX ]]; then
   echo "please, setup Xilinx environment [\"setXilinx\"]!"       
else
    echo "Found Xilinx environment"
    echo ""
    echo "programming [$board] FPGA at position [$pos] with the file: [$bitfile]" 
   # if [[ $board == "ML605" ]]; then 
   #     printf "setMode -bs \nsetCable -port auto \nsetCableSpeed -speed 6000000 \nidentify \nassignFile -p $pos -file $bitfile \nprogram -p $pos \nquit\"" >> batch_temp.cmd

   # elif [[ $board == "FECv6" ]]; then 
   #     printf "setMode -bs \nsetCable -port auto \nsetCableSpeed -speed 6000000 \nidentify \nassignFile -p $pos -file $bitfile \nprogram -p $pos \nquit\"" >> batch_temp.cmd

   # else
   #     echo "Specify PCB type to be either [FECv6] or [ML605] as second argument!"
   #     echo "Using ML605 file type..." 
   #     printf "setMode -bs \nsetMode -bs \nsetCable -port auto \nidentify -inferir \nidentifyMPM \nassignFile -p $pos -file $bitfile \nprogram -p $pos \nquit\"" >> batch_temp.cmd

   # fi
    printf "setMode -bs \nsetMode -bs \nsetCable -port auto \nidentify -inferir \nidentifyMPM \nassignFile -p $pos -file $bitfile \nprogram -p $pos \nsetMode -bs \nquit\"" >> batch_temp.cmd
    cat batch_temp.cmd
    echo""
    echo "Running impact in batch command mode..."
    echo ""
    sleep 2
    #impact -batch batch_temp.cmd
    echo ""
    echo "Done!"
fi 

