// written by Jongsoo Kim
// Last modification: 2014-06-22
//
// compilation with NVIDIA HPC SDK
// nvc++ -o matmul_host -acc=host -fast -Minfo=opt matmul.c 
// nvc++ -o matmul_multicore -acc=multicore -fast -Minfo=opt matmul.c
// nvc++ -o matmul_gpu -acc=gpu -fast -Minfo=opt matmul.c
// revision history:  
// 20210711 JK added openACC directives

// N1024 : Intel(R) Xeon(R) CPU           E5640  @ 2.67GHz
// elapsed time in seconds = 9.750000e+00
// number of multiplications and additions  = 2.147484e+09
// 2.202547e-01 Gflops

// N=1024, Intel(R) Xeon(R) CPU E5-2660 @ 2.20GHz
// elapsed time in seconds = 2.080000e+00
// number of multiplications and additions  = 2.147484e+09
// 1.032444e+00 Gflops

#include <stdio.h>  
#include <stdlib.h>  // for malloc
#include <time.h>    // for the clock() function

const int N = 4096;

void MatMul(const float * A, const float * B, float * C) {
    
    #pragma acc kernels
//    #pragma acc data copyin (A[0:N*N],B[0:N*N]) copyout (C[0:N*N]) if(accelerate)
    for (int row=0; row<N; ++row)  
    for (int col=0; col<N; ++col) { 
        float Cvalue = 0;
        for (int k = 0; k < N; ++k) {
            Cvalue += A[row * N + k] * B[k * N + col];
        }
        C[row*N+col] = Cvalue;
    }

}

int main (void) {


    float * A, * B, *C;
    size_t size = N * N * sizeof(float);
    A = (float *) malloc(size);
    B = (float *) malloc(size);
    C = (float *) malloc(size);

    for (int row=0; row<N; ++row)  
    for (int col=0; col<N; ++col) { 
        A[row * N + col] = row + col + 1.0f; 
        B[row * N + col] = row + col + 1.0f; 
    }

    double etime = (double) clock(); // start to measure the elapsed time

    MatMul (A, B, C);

    etime = ((double) clock() - etime)/(double)CLOCKS_PER_SEC ;  // elapsed time

//    for (int row=0; row<N; ++row)  
//    for (int col=0; col<N; ++col) { 
//        printf("%f\n", C[row*N+col]);
//    }

    printf("elapsed time in seconds = %e\n",etime);
    double num_ops = 2.0f*N*N*N;
    printf("number of multiplications and additions  = %e\n",num_ops);
    printf("%e Gflops\n",num_ops/etime/1.e9);

    free(A); free(B); free(C);

    return 0;
}
