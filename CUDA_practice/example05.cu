#include <iostream>
#include <cuda.h>
#include <time.h>

using namespace std;

// Matrix multiplication process

// serial code
void matmul(int *a, int *b, int *c, int M, int N)
{
    int cal = 0;
    int idx_a;
    int idx_b;
    int idx_c;

    for(int row = 0; row < M; row ++){
        for(int col = 0; col < N; col++){
            cal = 0;
            for(int k = 0; k < M; k++){
                idx_a = row * N + k;
                idx_b = k * N + col;
                cal += a[idx_a] * b[idx_b];
            }
            idx_c = row * N + col;
            c[idx_c] = cal;
        }
    }
}

// parallel code : global memory
__global__ void parallel_matmul(int *a, int *b, int *c, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int cal = 0;

    for(int k = 0; k < M; k++){
        cal += a[row * N + col] * b[k*N + col];
    }

    c[row * N + col] = cal;
}

int main(int argc, char *argv[])
{
    int N = 4096;
    int BLOCK_DIMS = 128;
    size_t size = N * N * sizeof(int);

    // matrix pointer from cpu
    int *a = (int *)malloc(size);
    int *b = (int *)malloc(size);
    int *c = (int *)malloc(size);

    // initialize
    for (int i = 0; i < N * N; i++)
    {
        a[i] = (int)pow(-1, i);
        b[i] = (int)pow(-1, i-1);
    }

    // matrix pointer from gpu
    int *dev_a, *dev_b, *dev_c;

    // allocate memory from cpu to gpu
    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

    // measure running time (unit : ms)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // memory copy : from cpu to gpu
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_DIMS, BLOCK_DIMS);
    dim3 dimGrid(N / dimBlock.x, N/dimBlock.y);

    parallel_matmul<<<dimGrid,dimBlock>>>(dev_a, dev_b, dev_c, N, N);

    // memory copy : from gpu to cpu
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // time event end
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float eTime;
    cudaEventElapsedTime(&eTime, start, stop);
    eTime /= 1000;

    // barrier
    cudaDeviceSynchronize();

    printf("eTime for matrix multiplication : %e\n", eTime);

    double num_ops = 2.0f * N * N * N;
    printf("number of multiplications and additions  = %e\n", num_ops);
    printf("%e Gflops\n", num_ops / eTime / 1.e9);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);

    return 0;
}