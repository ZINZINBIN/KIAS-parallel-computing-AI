// written by Jongsoo Kim
// last modification: 2010-12-16
// Formatted output is only supported from computer capability higher than 2.x.
// compile options nvcc -arch sm_70 hello.cu

#include <stdio.h>

__global__ void hello_world(void) {
    printf("Hello World\n");
}

int main (void) {
    hello_world<<<1,5>>>();
    cudaDeviceSynchronize();
    return 0;
}
