// written by Jongsoo Kim
// Last modification: 2014-06-04 
// compile options nvcc -arch sm_20 exec_conf_1D.cu

#include <stdio.h>

__global__ void exec_conf(void) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    printf("tid = %d, threadIdx = (%d,%d,%d), blockIdx = (%d,%d,%d), blockDim = (%d,%d,%d), gridDim = (%d,%d,%d)\n", 
            tid, 
            threadIdx.x,threadIdx.y,threadIdx.z,
            blockIdx.x,blockIdx.y,blockIdx.z,
            blockDim.x,blockDim.y,blockDim.z,
            gridDim.x,gridDim.y,gridDim.z);

}

int main (void) {

    exec_conf<<<2,4>>>();
    cudaDeviceSynchronize();
    return 0;
}
