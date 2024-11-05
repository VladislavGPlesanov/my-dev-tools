
for i in {1..10}
do
    echo " "
done
echo " ================ [COMPILING] ================"
echo " "

##############################################################################
NUMPY_VER=$(python3 -c 'import numpy; print(numpy.__version__)')
echo "found numpy version $NUMPY_VER"
NUMPY_INCLUDE=$(python3 -c 'import numpy; print(numpy.get_include())') 
NUMPY_LIB=$(python3 -c 'import numpy; print(numpy.__file__.rsplit("/",1)[0] + "/core/lib")')

echo $NUMPY_INCLUDE
echo $NUMPY_LIB
##############################################################################
g++ myReadoutBuffer.cpp \
-o ArraySorter.so \
-shared -std=c++14 \
-fPIC \
-fpermissive \
-I $NUMPY_INCLUDE \
-L $NUMPY_LIB
##############################################################################
echo " "
echo " ================ [END] ================"
for i in {1..10}
do
    echo " "
done
#
#-I /home/vlad/readoutSW/miniconda/pkgs/numpy-1.26.2-py311h64a7726_0/lib/python3.11/site-packages/numpy/core/include/ \
#-L /home/vlad/readoutSW/miniconda/pkgs/numpy-1.26.2-py311h64a7726_0/lib/python3.11/site-packages/numpy/core/lib

#-I /home/vlad/readoutSW/miniconda/lib/python3.11/site-packages/numpy/core/include/ \
#-L /home/vlad/readoutSW/miniconda/lib/python3.11/site-packages/numpy/core/lib/
#switch between -shared -std=c++14/11 \
