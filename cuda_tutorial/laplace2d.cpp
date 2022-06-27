// programmed by Jongsoo Kim 
// last modification: 2021-07-18
//
// solving the Laplace equation, nable T = 0 
// with a 2D retangular domain.
//
// compilation options
// nvc++ -o laplace2d_host -acc=host -fast -Minfo=opt laplace2d.cpp 
// nvc++ -o laplace2d_gpu -acc=gpu -gpu=cc80,time -fast -Minfo=accel laplace2d.cpp 
//
// to select a device from multiple devices, use the following environment 
// export CUDA_VISIBLE_DEVICES=2

#include <iostream>
#include <cmath>
#include <chrono>  // for measuring execution time
using namespace std::chrono;

int main(int argc, char** argv)
{
    const int n = 4096; 
    const int m = 4096;
    const int iter_max = 1000;
    
    const float tol = 1.0e-5f;
    float error     = 1.0f;
    
    // put "static" to avoid "segmentation fault" error, if the array size is larger
    // than 512x512
    static float A[m][n] = {0.0f};   // initialize with a zero value
    static float Anew[m][n];         // no need to initialization

    
    // set boundary conditions
    for (int j = 0; j < n; j++)
    {
        A[0][j]   = 0.f;
        A[m-1][j] = 1.f;
    }
    
    for (int i = 0; i < m; i++)
    {
        A[i][0] = 0.f;
        A[i][n-1] = 1.f;
    }
    
    std::cout << "Solving Laplace equation with the Jacobi relaxation method in a 2D mesh (" << m << "," << n << ")" << std::endl;
    
    // start time
    auto start = high_resolution_clock::now();

    int iter = 0;
    
    #pragma acc data copy(A) create(Anew)
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

    #pragma acc kernels
    {
        #pragma acc loop tile(32,4) device_type(nvidia)
//        #pragma acc loop gang(32), vector(16)
        for( int i = 1; i < m-1; i++)
        {
//            #pragma acc loop gang(32), vector(16)
            for( int j = 1; j < n-1; j++ )
            {
                Anew[i][j] = 0.25f * ( A[i][j+1] + A[i][j-1]
                                     + A[i-1][j] + A[i+1][j]);
                error = fmaxf( error, fabsf(Anew[i][j]-A[i][j]));
            }
        }
        #pragma acc loop tile(32,4) device_type(nvidia)
//        #pragma acc loop gang(32), vector(16)
        for( int i = 1; i < m-1; i++)
        {
//            #pragma acc loop gang(32), vector(16)
            for( int j = 1; j < n-1; j++ )
            {
                A[i][j] = Anew[i][j];    
            }
        }

    }
//        if(iter % 100 == 0) 
//           std::cout << "iter=" << iter << " error=" << error << std::endl;
        
        iter++;
    }

    // end time
    auto end = high_resolution_clock::now();
    //
    // the required to call saxpy
    auto duration = duration_cast<microseconds>(end-start);
    std::cout << "time taken=" << duration.count() << "  microseconds" << std::endl;

}
