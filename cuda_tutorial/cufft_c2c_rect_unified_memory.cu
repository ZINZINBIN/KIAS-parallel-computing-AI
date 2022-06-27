// DFT of a symmetric rectangular function
// Since the input function is a real and symmetric function,
// the DFT of this function has only real part.
// written by Jongsoo Kim, 2016-06-18
// nvcc -arch sm_52 -l cufft cufft_C2C_rect_unified_memory.cu
// The uniformed memory is not working for the cufft.

#include <stdio.h>
#include <cufft.h>

#define N 64 

int main(int argc, char **argv)
{
    // pointers for input and output files
    FILE *in_file, *out_file;

    // array for the in-place transform
    cufftComplex *data;
    cudaMallocManaged (&data, N*sizeof(cufftComplex));

    // initialization of input data
    for (int i = 0; i < N; ++i)
    {
        data[i].x = 0.0f;  // real part
        data[i].y = 0.0f;  // imaginary part
    }
    // data are symmetric with respect to i=0
    // set the unit values for the first six points 
    for (int i = 0; i < 6; ++i) data[i].x = 1.0f;
    // set the unit values for the last five points 
    for (int i = N-5; i < N; ++i) data[i].x = 1.0f;

    in_file = fopen("input.dat","w+");
    for (int i=0; i<N; i++) fprintf(in_file,"%d %e %e\n", i, data[i].x, data[i].y);
    fclose(in_file);
 
    cudaDeviceSynchronize();

    cufftHandle plan; 
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, data, data, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    out_file = fopen("output.dat","w+");
    for (int i=0; i<N; i++) fprintf(out_file,"%d %e %e\n", i, data[i].x, data[i].y);
    fclose(out_file);

    cufftDestroy(plan); 
    cudaFree(data);
}
