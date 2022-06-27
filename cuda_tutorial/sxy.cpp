// compilation 
//   nvc++ -o sxy_host -acc=host -fast -Minfo=opt sxy.cpp 
//   nvc++ -o sxy_multicore -acc=multicore -fast -Minfo=opt sxy.cpp 
//   nvc++ -o sxy_gpu -acc=gpu -gpu=cc70,time -fast -Minfo=accel sxy.cpp  
//
// to select a device from multiple devices, use the following environment 
// export CUDA_VISIBLE_DEVICES=2
//
// revision history:  
//   20210711 JK write a c++ program without an openACC compiler directive

#include <iostream>
#include <chrono>  // for measuring execution time
using namespace std::chrono;

void sxy(const float *x, const float *y, float *z, const unsigned int n) 
{ 
//    #pragma acc kernels loop independent copyin(x[0:n],y[0:n]),copyout(z[0:n])
#pragma acc kernels
{
    #pragma acc loop independent 
    for (int i=0; i<n; i++) 
    {
        z[i] = x[i] + y[i];
    }
}
}
     
int main (void) 
{
    const unsigned int n = 1 << 20;  // power of 2

    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];

    for (int i=0; i<n; i++) 
    {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }

    // start time
    auto start = high_resolution_clock::now();

    sxy (x, y, z, n);

    // end time
    auto end = high_resolution_clock::now();

     // the required to call saxpy
    auto duration = duration_cast<microseconds>(end-start);
    std::cout << "time taken by sxy function  " << duration.count() << "  microseconds" << std::endl;
    std::cout << n/duration.count() << "  MFLOPS" << std::endl;

    // write the results of the first and last vector sums
    std::cout << "first element: " << x[0] << "+" << y[0] << "=" << z[0] << std::endl;
    std::cout << "last element: " << x[n-1] << "+" << y[n-1] << "=" << z[n-1] << std::endl;

    delete [] x;
    delete [] y;
    delete [] z;

    return 0;
}
