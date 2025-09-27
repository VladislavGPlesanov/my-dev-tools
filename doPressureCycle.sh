ncycles=$1
twait=$2

for ((i=1; i<=$1; i++))  
do
    echo "iteration $i"
    echo "waiting $2 seconds"
    pid=$(python3 plotPolygons.py)
    sleep $2
    echo "done with a step"
done


############### template for bash script on tpc08 ##################
#
#echo "Starting $1 pressure cycles with waiting time of $2"
#sleep 2 
#for ((i=1; i<=$1; i++))
#do
#    echo "ramp-up nr $i\n"
#    ./pressureControl -t /dev/ttyUSB0/ -p 1.5 -l
#    echo "ramp-up done!\n"
#    echo "waiting 10 min"
#    sleep 600
#    echo "going down!\n"
#    ./pressureControl -t /dev/ttyUSB0/ -p 0.0 -l
#    echo "pressure released!\n"
#    sleep 60
#
#done

