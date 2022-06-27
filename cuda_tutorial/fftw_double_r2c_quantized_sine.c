// discrete fourier transform of a sine wave 
// Since the sine wave is a odd function, its fourier transform has only imaginary part
// written by Jongsoo Kim, 2014-06-29
// gcc -std=c99 -I/usr/include -L/usr/lib64 -lm -lfftw3 fftw_double_r2c_quantized_sine.c

#include <stdio.h>
#include <math.h>
#include <fftw3.h>

#define TWOPI 6.28318530717959
#define N 64

int main(void)
{
    FILE *in_file, *out_file;

    // the size of input array has n 
    double *in;
    in = (double *) fftw_malloc(sizeof(double) * N);

    // the size of output array should have n/2+1
    fftw_complex *out;
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1));

    // initialization of input data
    for (int i = 0; i < N; ++i) in[i] = floor(7.05*sin(TWOPI*(double) i/(double) N)+0.5);

    in_file = fopen("fftw_double_r2c_quantized_sine_in.dat","w+");
    for (int i=0; i<N; i++) fprintf(in_file,"%d %e\n", i, in[i]);
    fclose(in_file);

    fftw_plan plan;
    plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    out_file = fopen("fftw_double_r2c_quantized_sine_out.dat","w+");
    for (int i=0; i<N/2+1; i++) fprintf(out_file,"%d %e \n", i, 20.0*log10(fabs(out[i][1]/out[1][1])));

    fftw_destroy_plan(plan);
    fftw_free(in); fftw_free(out);

    return 0;
}
