export CPLUS_INCLUDE_PATH=/usr/include/python3.5m
# Reference: https://github.com/BVLC/caffe/issues/410
# This is why we need to add this line

rm -r ./build/*
cd build
cmake ..
make