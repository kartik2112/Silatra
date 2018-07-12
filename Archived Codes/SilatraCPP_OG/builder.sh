export CPLUS_INCLUDE_PATH=/usr/include/python3.5m
# Reference: https://github.com/BVLC/caffe/issues/410
# This is why we need to add this line
# This is used for embedding Python in C++ and making it work at low level
# The file that needs it as of 3 Dec 2017 is `./GetMyHand/PythonScriptImagePassingHandlers/skinColorSegmentation.cpp``

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
# Reference: https://stackoverflow.com/a/43143994/5370202

rm -r ./build/*
cd build
cmake ..
make
#if [ $1 == "-cap" ]
#then echo $*
#fi
# ./SiLaTra
