export CPLUS_INCLUDE_PATH=/usr/include/python3.5m
# Reference: https://github.com/BVLC/caffe/issues/410
# This is why we need to add this line

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
# Reference: https://stackoverflow.com/a/43143994/5370202

rm -r ./build/*
cd build
cmake ..
make