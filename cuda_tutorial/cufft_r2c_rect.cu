// DFT of a symmetric rectangular function
// written by Jongsoo Kim, 2016-06-18
// nvcc -l cufft cufft_r2c_rect.cu

#include <stdio.h>
#include <cufft.h>

#define N 64 

int main(int argc, char **argv)
{
    FILE *in_file, *out_file;

    // define in and out pointers in host memory
    cufftReal    *h_in = (cufftReal *)malloc(sizeof(cufftReal) * N);
    cufftComplex *h_out = (cufftComplex *)malloc(sizeof(cufftComplex) * (N/2+1));

    // define in and out pointers in device memory
    cufftReal    *d_in;
    cufftComplex *d_out;
    cudaMalloc( (void **)&d_in, sizeof(cufftReal) * N);
    cudaMalloc( (void **)&d_out,sizeof(cufftComplex) * (N/2+1));

    // initialization of input data
    for (int i = 0; i < N; ++i)
    {
        h_in[i] = 0.0f;  // real part
    }

    // data are symmetric with respect to i=0
    // set the unit values for the first six points
    for (int i = 0; i < 6; ++i) h_in[i] = 1.0f;

    // set the unit values for the last five points
    for (int i = N-5; i < N; ++i) h_in[i] = 1.0f;

    in_file = fopen("cufft_r2c_rect_in.dat","w+");
    for (int i=0; i<N; i++) fprintf(in_file,"%d %e \n", i, h_in[i]);
    fclose(in_file);

    cudaMemcpy(d_in, h_in, sizeof(cufftReal)*N, cudaMemcpyHostToDevice);

    cufftHandle plan; 
    cufftPlan1d(&plan, N, CUFFT_R2C, 1);
    cufftExecR2C(plan, d_in, d_out);
    cudaMemcpy(h_out, d_out, sizeof(cufftComplex) * (N/2+1), cudaMemcpyDeviceToHost);

    out_file = fopen("cufft_r2c_rect_out.dat","w+");
    for (int i=0; i<N/2+1; i++) fprintf(out_file,"%d %e %e\n", i, h_out[i].x, h_out[i].y);
    fclose(out_file);

    cufftDestroy(plan); 
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
}
