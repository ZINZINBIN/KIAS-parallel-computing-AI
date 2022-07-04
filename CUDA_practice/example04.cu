#include <iostream>
#include <cuda.h>

using namespace std;

// Reduction process : dot operation

// serial code
void dot(int *a, int *b, int *c, int N)
{
    for (int i = 0; i < N; i++)
    {
        c += a[i] * b[i];
    }
}

// parallel code
__global__ void parallel_add(int *a, int *b, int *c, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char *argv[])
{
    int N = 4 * 16 * 16;
    int G = 4;
    int M = 16;
    int T = 16;

    int *a = (int *)malloc(N * sizeof(int));
    int *b = (int *)malloc(N * sizeof(int));
    int *c = (int *)malloc(N * sizeof(int));

    int *dev_a, *dev_b, *dev_c;

    // allocate memory from cpu to gpu
    cudaMalloc(&dev_a, N * sizeof(int));
    cudaMalloc(&dev_b, N * sizeof(int));
    cudaMalloc(&dev_c, N * sizeof(int));

    // cudaMallocManged : unified memory(cpu와 gpu 동시 접근 가능)에 할당하는 경우 cpu와 gpu에 각각 malloc 할 필요 없음

    // initialize
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // memory copy : from cpu to gpu
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grids(G, 1, 1);
    dim3 blocks(M, 1, 1);
    dim3 threads(T, 1, 1);

    parallel_add<<<blocks, threads>>>(dev_a, dev_b, dev_c, N);

    // memory copy : from gpu to cpu
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // barrier
    cudaDeviceSynchronize();

    // display the result
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);

    return 0;
}