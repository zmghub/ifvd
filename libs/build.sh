#!/bin/bash

# Configuration
CUDA_GENCODE="\
-gencode=arch=compute_70,code=sm_70" 


cd src
nvcc -I/usr/local/cuda/include --expt-extended-lambda -O3 -c -o bn.o bn.cu -x cu -Xcompiler -fPIC -std=c++11 ${CUDA_GENCODE}
cd ..
