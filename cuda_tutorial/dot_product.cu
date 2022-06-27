// last modification: 2011-06-28
// compile options nvcc dot_product.cu

#include <stdio.h>
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 4096;
const int threadsPerBlock = 1024;
const int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock; 

__global__ void dot_product(float *a, float *b, float *partial_c) { 

    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // set the cache values
    cache[threadIdx.x] = a[tid] * b[tid];

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock (blockDim.x) must be a power of 2
    // becuase of the following code

    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x == 0) partial_c[blockIdx.x] = cache[0];

}

int main (void) { 

    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the CPU side
    a = (float*) malloc ( N*sizeof(float) );
    b = (float*) malloc ( N*sizeof(float) );
    partial_c = (float*) malloc ( blocksPerGrid*sizeof(float) );

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N*sizeof(float) );
    cudaMalloc( (void**)&dev_b, N*sizeof(float) );
    cudaMalloc( (void**)&dev_partial_c, blocksPerGrid*sizeof(float) );

    for (int i=0; i<N; i++) {
       a[i] = (float) i;
       b[i] = (float) i;
    }

    cudaMemcpy ( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy ( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice );

    dot_product<<<blocksPerGrid,threadsPerBlock>>>(dev_a,dev_b,dev_partial_c);

    cudaMemcpy ( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost );

    c = 0.0f;
    for (int i=0; i<blocksPerGrid; i++)
        c += partial_c[i];

    printf("Dot prodoct of a and b = %f\n", c);
    printf("sum_squares of (N-1) = %f\n", sum_squares((float)(N-1)) );

    // free memory on the CUP side
    free(a); free(b); free(partial_c);

    // free memory on the GPU side
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_partial_c);

    return 0;
}
