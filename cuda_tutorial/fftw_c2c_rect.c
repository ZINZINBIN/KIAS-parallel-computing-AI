// discrete fourier transform of a retangular function
// written by Jongsoo Kim, 2014-06-29
// gcc -std=c99 -I/usr/include -L/usr/lib64 -lfftw3f fftw_c2c_rect.c

#include <stdio.h>
#include <fftw3.h>

#define N 64

int main(void)
{
    FILE *in_file, *out_file;

    // Since the in-place transform will be used, only one array for the fft
    // is enough. But it is usually more efficient to use the out-of-place transform. 
    fftwf_complex *data;
    data = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N);

    // initialization of input data
    for (int i = 0; i < N; ++i)
    {
        data[i][0] = 0.0f;  // real part
        data[i][1] = 0.0f;  // imaginary part
    }

    // data are symmetric with respect to i=0
    // set the unit values for the first six points 
    for (int i = 0; i < 6; ++i) data[i][0] = 1.0f;

    // set the unit values for the last five points 
    for (int i = N-5; i < N; ++i) data[i][0] = 1.0f;

    in_file = fopen("fftw_c2c_rect_in.dat","w+");
    for (int i=0; i<N; i++) fprintf(in_file,"%d %e %e\n", i, data[i][0], data[i][1]);
    fclose(in_file);

    // plan should go before initialization
    fftwf_plan plan;
    plan = fftwf_plan_dft_1d(N, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan); 

    out_file = fopen("fftw_c2c_rect_out.dat","w+");
    for (int i=0; i<N/2+1; i++) fprintf(out_file,"%d %e %e\n", i, data[i][0], data[i][1]);

    fftwf_destroy_plan(plan);
    fftwf_free(data);

    return 0;
}
