
for i in {1..10}
do
    echo " "
done
echo " ================ [COMPILING] ================"
echo " "
##############################################################################
g++ myReadoutBuffer.cpp \
-o ArraySorter.so \
-shared -std=c++14 \
-fPIC \
-fpermissive \
-I /home/vlad/readoutSW/miniconda/pkgs/numpy-1.26.2-py311h64a7726_0/lib/python3.11/site-packages/numpy/core/include/ \
-L /home/vlad/readoutSW/miniconda/pkgs/numpy-1.26.2-py311h64a7726_0/lib/python3.11/site-packages/numpy/core/lib
##############################################################################
echo " "
echo " ================ [END] ================"
for i in {1..10}
do
    echo " "
done
#-I /home/vlad/readoutSW/miniconda/lib/python3.11/site-packages/numpy/core/include/ \
#-L /home/vlad/readoutSW/miniconda/lib/python3.11/site-packages/numpy/core/lib/
#switch between -shared -std=c++14/11 \
