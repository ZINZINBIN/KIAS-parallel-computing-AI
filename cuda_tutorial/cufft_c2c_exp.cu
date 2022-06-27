// DFT of a symmetric rectangular function
// written by Jongsoo Kim, 2021-09-22
// nvcc -l cufft -arch=sm_70 cufft_c2c_exp.cu

#include <stdio.h>
#include <cufft.h>

#define N 32 

int main(int argc, char **argv)
{
    FILE *in_file, *out_file;
    // size of memory space for input and output data
    int mem_size = sizeof(cufftComplex) * N;

    // define data in host memory
    cufftComplex *h_data = (cufftComplex *)malloc(mem_size);

    // Since the in-place transform will be used, only one array for the fft    
    // is enough. But it is usually more efficient to use the out-of-place transform. 
    cufftComplex *d_data;
    cudaMalloc( (void **)&d_data,mem_size );

    float T = 0.25f;
    // initialization of input data
    h_data[0].x = 0.5f;
    h_data[0].y = 0.0f;  // imaginary part

    for (int i = 1; i < N; ++i)
    {
        h_data[i].x = exp(-T*i);
        h_data[i].y = 0.0f;  // imaginary part
    }

    in_file = fopen("cufft_c2c_exp_in.dat","w+");
    for (int i=0; i<N; i++) fprintf(in_file,"%d %e %e\n", i, h_data[i].x, h_data[i].y);
    fclose(in_file);
 
    cudaMemcpy(d_data, h_data, mem_size, cudaMemcpyHostToDevice);

    cufftHandle plan; 
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaMemcpy(h_data, d_data, mem_size, cudaMemcpyDeviceToHost);

    out_file = fopen("cufft_c2c_exp_out.dat","w+");
    for (int i=0; i<N; i++) fprintf(out_file,"%d %e %e\n", i, T*h_data[i].x, T*h_data[i].y);
    fclose(out_file);

    cufftDestroy(plan); 
    free(h_data);
    cudaFree(d_data);
}
