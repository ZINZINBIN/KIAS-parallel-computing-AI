#include <iostream>
#include <cuda.h>

using namespace std;

// callable only on cpu
// executed from device
__global__ void func_cpu(void)
{
    printf("executed from device\n");
}

// callable only on gpu and executed from device only
__device__ void func_device(){
    printf("executed from device only\n");
}

// built-in variables : threadIdx, blockIdx, blockDim, gridDim
// built-in vector type : dim3

// example : 1D execution configuration 
__global__ void exec_conf(void){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("tid : %d, threadIdx : (%d,%d,%d), blockIdx : (%d,%d,%d), blockDim : (%d,%d,%d), gridDim : (%d,%d,%d)\n",
    tid,
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z    
    );
}

__global__ void exec_conf_2D(void)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    printf("id_x : %d, id_y : %d, threadIdx : (%d,%d,%d), blockIdx : (%d,%d,%d), blockDim : (%d,%d,%d), gridDim : (%d,%d,%d)\n",
           id_x, id_y,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

__global__ void exec_conf_3D(void)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    printf("id_x : %d, id_y : %d, id_z : %d, threadIdx : (%d,%d,%d), blockIdx : (%d,%d,%d), blockDim : (%d,%d,%d), gridDim : (%d,%d,%d)\n",
           id_x, id_y, id_z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char *argv[])
{
    /* Execution configuration
     * (ex) kernel<<< # of block per grid, # of threads per block >>>(args)
     */

    int n_block = 2;
    int n_thread = 4;

    printf("\n # execution configuration for 1D \n");

    exec_conf<<<n_block, n_thread>>>();
    cudaDeviceSynchronize();

    printf("\n # execution configuration for 2D \n");

    dim3 blocks(2,2,1);
    dim3 threads(2,2,1);

    exec_conf_2D <<<blocks, threads>>>();
    cudaDeviceSynchronize();

    // cudaDeviceReset();
    return 0;
}