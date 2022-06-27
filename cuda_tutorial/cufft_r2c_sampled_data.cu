// written by Jongsoo Kim
// last modification: June 24, 2016
// nvcc -l cufft cufft_r2c_sampled_data.cu
// this program is to show how cufft works in a batch mode. 
// the calculation of power spectrum should be implemented. 

#include <stdio.h>
#include <math.h>
#include <cufft.h>
#include <time.h>    // for the clock() function
#define TWOPI 6.28318530717959 

__global__ void power_spectrum(cufftComplex *, cufftReal *, size_t);

const int N=4096;
const int BATCH_SIZE = 10000;
const float fs = 10000.0;              /* sampling frequency [Hz] */

void samples_gen (float *samples)
{
    float f1 = 1234.0;             /* first signal frequency [Hz] */
    float amp1 = 2.82842712474619; /* 2 Vrms */
    float f2 = 2500.2157;          /* second signal frequency [Hz] */
    float amp2 = 1.0;              /* 0.707 Vrms */
    float ulsb = 1.0e-3;           /* Value of 1 LSB in Volt */
    float t, u;

    for (size_t i = 0; i < N*BATCH_SIZE; i++)
    {
        t = (float) i / fs;
        u = amp1 * sin (TWOPI * f1 * t) + amp2 * sin (TWOPI * f2 * t);
        samples[i] = floor (u / ulsb + 0.5) * ulsb; /* Rounding */
    }
}

int main (int argc, char *argv[])
{
    FILE *out_file;

    int OUTPUT_SIZE = N/2+1;
    float fres = fs/(float) N;

    float *h_in;
    h_in = (float *)malloc(sizeof(float) * N * BATCH_SIZE);
    cufftComplex *h_out;
    h_out = (cufftComplex *)malloc(sizeof(cufftComplex) * OUTPUT_SIZE * BATCH_SIZE);
    float *ps;
    ps = (float *)malloc(sizeof(float) * OUTPUT_SIZE);
    for (int i=0;i<OUTPUT_SIZE;++i) ps[i] = 0.0f;

    // generate input data
    samples_gen (h_in);

    double etime = (double) clock(); // start to measure the elapsed time

    float *d_in;  
    cufftComplex *d_out; 

    cudaMalloc( (void **)&d_in, sizeof(float) * N * BATCH_SIZE);
    cudaMalloc( (void **)&d_out,sizeof(cufftComplex) * OUTPUT_SIZE * BATCH_SIZE);

    cudaMemcpy(d_in, h_in, sizeof(float) * N * BATCH_SIZE, cudaMemcpyHostToDevice);

    // make a plan for fft
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_R2C, BATCH_SIZE);
    cufftExecR2C(plan, d_in, d_out);

    cudaMemcpy(h_out, d_out, sizeof(cufftComplex) * OUTPUT_SIZE * BATCH_SIZE, cudaMemcpyDeviceToHost);

    etime = ((double) clock() - etime)/(double)CLOCKS_PER_SEC ;

    for (int j=0; j<BATCH_SIZE; j++) 
    { 
       for (int i=0;i<OUTPUT_SIZE;++i) 
       {   ps[i] += h_out[i+j*OUTPUT_SIZE].x*h_out[i+j*OUTPUT_SIZE].x + h_out[i+j*OUTPUT_SIZE].y*h_out[i+j*OUTPUT_SIZE].y;
       }
    }

    printf("elapsed time in seconds = %e\n",etime);
    double num_ops = BATCH_SIZE*5.0*(double) N * log10(N);
    printf("number of floating point operations = %e\n",num_ops);
    printf("%e Gflops\n",num_ops/etime/1.e9);

    out_file = fopen("cufft_r2c_sampled_data_out.dat","w+");
    for (int i=0;i<OUTPUT_SIZE;++i) fprintf(out_file,"%e %e \n", i*fres/1000.0, sqrt(2.0*ps[i]/(cufftReal)BATCH_SIZE/fs));

    cufftDestroy(plan);
    free(h_in); free(h_out); free(ps);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}
