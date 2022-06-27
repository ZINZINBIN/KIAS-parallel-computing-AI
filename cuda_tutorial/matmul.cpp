// written by Jongsoo Kim
// Last modification: 2014-06-22
//
// compilation with NVIDIA HPC SDK
// nvc++ -o matmul_host -acc=host -fast -Minfo=opt matmul.cpp 
// nvc++ -o matmul_multicore -acc=multicore -fast -Minfo=opt matmul.cpp
// nvc++ -o matmul_gpu -acc=gpu -gpu=cc80,time -fast -Minfo=accel matmul.cpp
//
// to select a device from multiple devices, use the following environment 
// export CUDA_VISIBLE_DEVICES=2
//
// revision history:  
// 20210711 JK wrote c++ version with openACC compiler directives

// AMD EPYC 7F32 8-Core Processor
// N=1024, matmul_host: 0.310939  Gflops
// A100-PCIE-40GB
// N=2048: matmul_gpu: 90.0767  Gflops 

#include <iostream>
#include <chrono>  // for measuring execution time
using namespace std::chrono;

const int N = 4096;

void MatMul(const float *restrict A, const float *restrict B, float *restrict C) {
    
#pragma acc kernels copyin (A[0:N*N],B[0:N*N]) copyout (C[0:N*N])
{   
   #pragma acc loop independent vector(16)
   for (int row=0; row<N; ++row) 
   { 
      #pragma acc loop independent vector(16)
      for (int col=0; col<N; ++col) 
      { 
         float Cvalue = 0;
         #pragma acc loop reduction(+:Cvalue)
         for (int k = 0; k < N; ++k) 
         {
            Cvalue += A[row * N + k] * B[k * N + col];
         }
         C[row*N+col] = Cvalue;
      }
   }

}

}

int main (void) {

    float *A, *B, *C;
    size_t size = N * N * sizeof(float);
    A = (float *) malloc(size);
    B = (float *) malloc(size);
    C = (float *) malloc(size);

    for (int row=0; row<N; ++row)  
    for (int col=0; col<N; ++col) { 
        A[row * N + col] = 1.0f; 
        B[row * N + col] = 1.0f; 
        C[row * N + col] = 0.0f; 
    }

    // start time
    auto start = high_resolution_clock::now();

    MatMul (A, B, C);

    // end time
    auto end = high_resolution_clock::now();

    // the required to call saxpy
    auto duration = duration_cast<microseconds>(end-start);

    double num_ops = 2.0f*N*N*N;

    std::cout << "time taken by MatMul function  " << duration.count() << "  microseconds" << std::endl;
    std::cout << num_ops/duration.count()/1000.0 << "  Gflops" << std::endl;

    // write the results of the first and last vector sums
    std::cout << "first element: " << C[0] << std::endl;
    std::cout << "last element: " << C[(N-1)*(N-1)] << std::endl;

    free(A); free(B); free(C);

    return 0;
}
