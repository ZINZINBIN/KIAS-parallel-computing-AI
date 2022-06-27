// discrete fourier transform of a retangular function
// written by Jongsoo Kim, 2014-06-29
// gcc -std=c99 -I/usr/include -L/usr/lib64 -lfftw3f fftw_r2c_rect.c

#include <stdio.h>
#include <fftw3.h>

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
    for (int i = 0; i < N; ++i) in[i] = 0.0f;

    // data are symmetric with respect to i=0
    // set the unit values for the first six points 
    for (int i = 0; i < 6; ++i) in[i] = 1.0f;

    // set the unit values for the last five points 
    for (int i = N-5; i < N; ++i) in[i] = 1.0f;

    in_file = fopen("fftw_r2c_rect_in.dat","w+");
    for (int i=0; i<N; i++) fprintf(in_file,"%d %e\n", i, in[i]);
    fclose(in_file);

    fftwf_plan plan;
    plan = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftwf_execute(plan);

    out_file = fopen("fftw_r2c_rect_out.dat","w+");
    for (int i=0; i<N/2+1; i++) fprintf(out_file,"%d %e %e\n", i, out[i][0], out[i][1]);

    fftwf_destroy_plan(plan);
    fftwf_free(in); fftwf_free(out);

    return 0;
}
