//declare constant memory
#include <stdio.h>
__constant__ float cangle[360];

__global__ void test_kernel(float* darray)
{
    int tid;
    
    //calculate each thread global index
    tid = blockIdx.x * blockDim.x + threadIdx.x;
   
    for(int loop=0;loop<360;loop++)
        darray[tid]= darray [tid] + cangle [loop] ;
    return;
}

int main(int argc,char** argv)
{
    int size=8192;
    float* darray;
    float hangle[360];

    //allocate device memory
    cudaMalloc ((void**)&darray,sizeof(float)*size);
         
    //initialize allocated memory
    cudaMemset (darray,0,sizeof(float)*size);

    //initialize angle array on host
    for(int loop=0;loop<360;loop++) {
        hangle[loop] = acos( -1.0f )* loop/ 180.0f;
//        printf("loop=%d, angle=%f\n", loop, hangle[loop]);
    }

    //copy host angle data to constant memory
    cudaMemcpyToSymbol (cangle, hangle, sizeof(float)*360  );
   
    test_kernel<<<size/64,64>>>(darray);
     
    //free device memory
    cudaFree(darray);

    return 0;
}
