// discrete fourier transform of a sine wave 
// Since the sine wave is a odd function, its fourier transform has only imaginary part
// written by Jongsoo Kim, 2014-06-29
// gcc -std=c99 -I/usr/include -L/usr/lib64 -lm -lfftw3f fftw_single_r2c_sine.c

#include <stdio.h>
#include <math.h>
#include <fftw3.h>

#define TWOPI 6.28318530717959
#define N 64

int main(void)
{
    FILE *in_file, *out_file;

    // the size of input array has n 
    float *in;
    in = (float *) fftwf_malloc(sizeof(float) * N);

    // the size of output array should have n/2+1
    fftwf_complex *out;
    out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (N/2+1));

    // initialization of input data
    for (int i = 0; i < N; ++i) in[i] = sin(TWOPI*(float) i/(float) N);

    in_file = fopen("fftw_single_r2c_sine_in.dat","w+");
    for (int i=0; i<N; i++) fprintf(in_file,"%d %e\n", i, in[i]);
    fclose(in_file);

    fftwf_plan plan;
    plan = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftwf_execute(plan);

    out_file = fopen("fftw_single_r2c_sine_out.dat","w+");
    for (int i=0; i<N/2+1; i++) fprintf(out_file,"%d %e %e \n", i, out[i][0],out[i][1]);
//    for (int i=0; i<N/2+1; i++) fprintf(out_file,"%d %e \n", i, 20.0*log(fabsf(out[i][1]/out[1][1])));

    fftwf_destroy_plan(plan);
    fftwf_free(in); fftwf_free(out);

    return 0;
}
