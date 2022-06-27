// DFT of a symmetric rectangular function
// written by Jongsoo Kim, 2016-06-18
// nvcc -l cufft cufft_c2c_rect.cu

#include <stdio.h>
#include <cufft.h>

#define N 64 

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

    // initialization of input data
    for (int i = 0; i < N; ++i)
    {
        h_data[i].x = 0.0f;  // real part
        h_data[i].y = 0.0f;  // imaginary part
    }

    // data are symmetric with respect to i=0
    // set the unit values for the first six points 
//    h_data[0].x = 0.5f;
    for (int i = 0; i < N/16+1; ++i) h_data[i].x = 1.0f;

    // set the unit values for the last five points 
    for (int i = 15*N/16; i < N; ++i) h_data[i].x = 1.0f;

    in_file = fopen("cufft_c2c_rect_in.dat","w+");
    for (int i=0; i<N; i++) fprintf(in_file,"%d %e %e\n", i, h_data[i].x, h_data[i].y);
    fclose(in_file);
 
    cudaMemcpy(d_data, h_data, mem_size, cudaMemcpyHostToDevice);

    cufftHandle plan; 
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaMemcpy(h_data, d_data, mem_size, cudaMemcpyDeviceToHost);

    out_file = fopen("cufft_c2c_rect_out.dat","w+");
    for (int i=0; i<N; i++) fprintf(out_file,"%d %e %e\n", i, h_data[i].x, h_data[i].y);
    fclose(out_file);

    cufftDestroy(plan); 
    free(h_data);
    cudaFree(d_data);
}
