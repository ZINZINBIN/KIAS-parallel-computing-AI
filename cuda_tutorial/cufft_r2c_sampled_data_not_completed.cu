// written by Jongsoo Kim
// last modification: June 24, 2016
// nvcc -l cufft cufft_r2c_sampled_data.cu
// reference: http://holometer.fnal.gov/GH_FFT.pdf

#include <stdio.h>
#include <math.h>
#include <cufft.h>
#include <time.h>    // for the clock() function
#define TWOPI 6.28318530717959 

__global__ void power_spectrum(cufftComplex *, cufftReal *, size_t);

const int N=4096;
const int BATCH_SIZE = 1;
const int OUTPUT_SIZE = N/2+1;
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

    float fres = fs/(float) N;

    cufftReal *h_in;
    h_in = (cufftReal *)malloc(sizeof(cufftReal) * N * BATCH_SIZE);
    cufftReal *h_ps;
    h_ps = (cufftReal *)malloc(sizeof(cufftReal) * OUTPUT_SIZE);

    // generate input data
    samples_gen (h_in);

    double etime = (double) clock(); // start to measure the elapsed time

    cufftReal *d_in;  
    cufftComplex *d_out; 
    cufftReal *d_ps; 

    cudaMalloc( (void **)&d_in, sizeof(cufftReal) * N * BATCH_SIZE);
    cudaMalloc( (void **)&d_out,sizeof(cufftComplex) * OUTPUT_SIZE * BATCH_SIZE);
    cudaMalloc( (void **)&d_ps, sizeof(cufftReal) * OUTPUT_SIZE );
    // initialize d_ps array
    cudaMemset( d_ps, 0.0f, sizeof(cufftReal) * OUTPUT_SIZE );

    cudaMemcpy(d_in, h_in, sizeof(cufftReal) * N * BATCH_SIZE, cudaMemcpyHostToDevice);

    // make a plan for fft
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_R2C, BATCH_SIZE);
    cufftExecR2C(plan, d_in, d_out);

    // calculation of power spectrum
    int numThreads = 128;
    int numGrids = (OUTPUT_SIZE*BATCH_SIZE + numThreads -1)/numThreads;
    printf("%d %d\n", numThreads, numGrids);
    power_spectrum<<<numGrids,numThreads>>>(d_out,d_ps,OUTPUT_SIZE);
    cudaMemcpy(h_ps, d_ps, sizeof(cufftReal)*OUTPUT_SIZE, cudaMemcpyDeviceToHost);

    etime = ((double) clock() - etime)/(double)CLOCKS_PER_SEC ;
    printf("elapsed time in seconds = %e\n",etime);
    double num_ops = BATCH_SIZE*5.0*(double) N * log10(N);
    printf("number of floating point operations = %e\n",num_ops);
    printf("%e Gflops\n",num_ops/etime/1.e9);

    out_file = fopen("cufft_r2c_sampled_data_out.dat","w+");
    for (int i=0;i<N/2+1;++i) fprintf(out_file,"%e %e \n", i*fres/1000.0, sqrt(2.0*h_ps[i]/(cufftReal)BATCH_SIZE/fs));

    cufftDestroy(plan);
    free(h_in); free(h_ps);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ps);

    return 0;
}

__global__ void power_spectrum(cufftComplex *c, cufftReal *a, size_t OUTPUT_SIZE)
{
    const size_t numThreads = blockDim.x * gridDim.x;
    const size_t threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = threadID; i < OUTPUT_SIZE*BATCH_SIZE ; i += numThreads)
    {
        size_t j = i % OUTPUT_SIZE;
        atomicAdd( &a[j], c[i].x*c[i].x+c[i].y*c[i].y);
    }
}
