// written by Jongsoo Kim, 2014-06-29
// nvcc -l cufft test.cu

#include <stdio.h>
#include <cufft.h>

#define NX 256 
#define BATCH 1 

typedef float2 Complex;

int main(int argc, char **argv)
{

// Allocate host memory for the signal
    Complex *h_signal = (Complex *)malloc(sizeof(Complex) * NX);
    Complex *f_signal = (Complex *)malloc(sizeof(Complex) * NX);
    Complex *d_signal;
    int mem_size = sizeof(Complex) * NX *BATCH;
    cudaMalloc( (void **)&d_signal,mem_size);

    // Initalize the memory for the signal
    for (unsigned int i = 0; i < NX/4; ++i)
    {
        h_signal[i].x = 1.0f;   
    }
    for (unsigned int i = NX/4; i < 3*NX/4; ++i)
    {
        h_signal[i].x = 0.0f;  
    }
    for (unsigned int i = 3*NX/4; i < NX; ++i)
    {
        h_signal[i].x = 1.0f;  
    }

    for (unsigned int i = 0; i < NX; ++i)
    {
        h_signal[i].y = 0.0f;
    }

    cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice);

//cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH); 
//if (cudaGetLastError() != cudaSuccess)
//{ 
//   fprintf(stderr, "Cuda error: Failed to allocate\n"); 
//   return 0;	 
//} 
cufftHandle plan; 
cufftPlan1d(&plan, NX, CUFFT_C2C, 1);
cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);
cudaMemcpy(f_signal, d_signal, mem_size, cudaMemcpyDeviceToHost);

    for (int i=0; i<NX; i++) {
        printf("(%e,%e)\n", f_signal[i].x,f_signal[i].y);
    }

//{ 
//   fprintf(stderr, "CUFFT error: Plan creation failed"); 
//   return;	 
//}	 

//... /* Note: * Identical pointers to input and output arrays implies in-place transformation */ 

//if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS)
//{ 
//   fprintf(stderr, "CUFFT error: ExecC2C Forward failed"); 
//   return 0;	 
//} 

//if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS)
//{ 
//   fprintf(stderr, "CUFFT error: ExecC2C Inverse failed"); 
//   return;	 
//} 

///* * Results may not be immediately available so block device until all * tasks have completed */ 

//if (cudaDeviceSynchronize() != cudaSuccess)
//{
//    fprintf(stderr, "Cuda error: Failed to synchronize\n"); 
//    return;	 
//}	 
///* * Divide by number of elements in data set to get back original data */ ... 
//cufftDestroy(plan); cudaFree(data);
}
