#include <iostream>
#include <cuda.h>

using namespace std;

__global__ void kernel(void){
    printf("kernel test\n");
}

int main(int argc, char *argv[]){

    kernel <<<1,5>>>();
    cudaDeviceReset();
    return 0;
}