// last modification: 2011-06-28
// compile options nvcc -arch=sm_12 histogram.cu 

#include <stdio.h>
#include <stdlib.h> // rand() function

const long SIZE = 100*1024*1024; 
const int threads = 256;
const int blocks = 128; 

__global__ void histo_kernel(unsigned char *buffer, unsigned int *histo) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    while (i < SIZE) {
        atomicAdd ( &(histo[buffer[i]]), 1 );
        i += blockDim.x * gridDim.x;
    }
}

int main (void) {

    unsigned char buffer[SIZE];

    for (int i=0; i<SIZE; i++)
        buffer[i] = rand()%256;

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start,0 );

    // allocate memory on the GPU
    unsigned char *dev_buffer;
    unsigned int *dev_histo;
    cudaMalloc( (void**)&dev_buffer, SIZE);
    cudaMemcpy( dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice );

    cudaMalloc( (void**)&dev_histo, 256*sizeof(int) );
    cudaMemset( dev_histo, 0, 256*sizeof(int) );

    histo_kernel<<<blocks,threads>>>(dev_buffer, dev_histo);

    unsigned int histo[256];
    cudaMemcpy( histo, dev_histo, 256*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    printf("elapsed time in seconds: %f\n", elapsedTime/1000.0 );

    long histoCount = 0;
    for (int i=0; i<256; i++)
        histoCount += histo[i];

    printf("SIZE: %ld\n", SIZE);
    printf("Histogram Sum: %ld\n", histoCount);

    for (int i=0; i<SIZE; i++)
        histo[buffer[i]]--;

    for (int i=0; i<256; i++)
        if (histo[i] !=0)
            printf( "Failure at %d!\n", i );

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    cudaFree( dev_histo );
    cudaFree( dev_buffer );
 
    return 0;
}
