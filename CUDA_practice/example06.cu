#include <iostream>
#include <cuda.h>
#include <time.h>

using namespace std;

// constant memory and texture memory

__constant__ float cangle[360];

__global__ void test_kernel(float *darr)
{
    int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int loop = 0; loop < 360; loop++){
        darr[tid] += cangle[loop];
    }
}

int main(int argc, char *argv[])
{
    int N = 1024 * 8;
    size_t size = N * sizeof(int);
    float *darr;
    float hangle[360];

    cudaMalloc(darr, size);



    return 0;
}