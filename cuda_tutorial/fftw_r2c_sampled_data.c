// written by Jongsoo Kim
// last modification: June 24, 2016
// gcc -std=c99 -I/usr/include -L/usr/lib64 -lm -lfftw3f fftw_r2c_sampled_data.c
// reference: http://holometer.fnal.gov/GH_FFT.pdf

#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>    // for the clock() function
#define TWOPI 6.28318530717959 

const int N=4096;
const int nave = 10000;
const float fs = 10000.0;              /* sampling frequency [Hz] */

void samples_gen (float *samples)
{
    float f1 = 1234.0;             /* first signal frequency [Hz] */
    float amp1 = 2.82842712474619; /* 2 Vrms */
    float f2 = 2500.2157;          /* second signal frequency [Hz] */
    float amp2 = 1.0;              /* 0.707 Vrms */
    float ulsb = 1.0e-3;           /* Value of 1 LSB in Volt */
    float t, u;

    for (size_t i = 0; i < N*nave; i++)
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

    float *samples;
    samples = (float*) fftwf_malloc(sizeof(float) * N*nave);
    samples_gen (samples);

    float *in;
    in = (float*) fftwf_malloc(sizeof(float) * N);

    // the output array should have N/2+1
    fftwf_complex *out;
    out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (N/2+1));

    // plan should go before initialization
    fftwf_plan plan;
    plan = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

    // array for an averaged power spectrum
    float *ps;
    ps = (float*) fftwf_malloc(sizeof(float) *(N/2+1));

    // initialization of ps
    for (int i=0;i<N/2+1;++i) ps[i] = 0.0;

    double etime = (double) clock(); // start to measure the elapsed time

    for (int j = 0; j < nave ; j++)
    {
        for (int i=0;i<N;++i) in[i] = samples[i+j*N];
        fftwf_execute(plan);
        for (int i=0;i<N/2+1;++i) ps[i] += out[i][0]*out[i][0] + out[i][1]*out[i][1];
    }

    etime = ((double) clock() - etime)/(double)CLOCKS_PER_SEC ;
    printf("elapsed time in seconds = %e\n",etime);
    double num_ops = nave*5.0*(double) N * log2(N);
    printf("number of floating point operations = %e\n",num_ops);
    printf("%e Gflops\n",num_ops/etime/1.e9);

    out_file = fopen("fftw_r2c_sampled_data_out.dat","w+");
    for (int i=0;i<N/2+1;++i) fprintf(out_file,"%e %e \n", i*fres/1000.0, sqrt(2.0*ps[i]/(float)nave/fs));

    fftwf_destroy_plan(plan);
    fftwf_free(in); fftwf_free(out);

    return 0;
}
