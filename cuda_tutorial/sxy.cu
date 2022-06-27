// written by Jongsoo Kim
// revision history:  
// 20210711 JK added #progma openACC compiler directive
// 
// nvcc -std=c++11 -o sxy sxy.cu 

#include <iostream>
#include <chrono>  // for measuring execution time
using namespace std::chrono;

__global__ void sxy(const float *x, const float *y, float *z, const unsigned int n) 
{ 
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // index for arrays 
    while (i < n) 
    {
       z[i] = x[i] + y[i];
       i += blockDim.x * gridDim.x;
    }
}
     
int main (void) 
{
    const unsigned int n = 1 << 20;  // power ofint 2

    // allocate the memory on host 
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];

    // allocate the memory on device 
    float *dev_x, *dev_y, *dev_z;
    cudaMalloc( &dev_x, n*sizeof(float) );
    cudaMalloc( &dev_y, n*sizeof(float) );
    cudaMalloc( &dev_z, n*sizeof(float) );

    for (int i=0; i<n; i++) 
    {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }

    // starting time
    auto start = high_resolution_clock::now();

    // copy the host arrays 'x' and 'y' to device 
    cudaMemcpy ( dev_x, x, n*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy ( dev_y, y, n*sizeof(float), cudaMemcpyHostToDevice );

    sxy<<<256,256>>>(dev_x, dev_y, dev_z, n);

    // copy the device array 'z' to host 
    cudaMemcpy ( z, dev_z, n*sizeof(float), cudaMemcpyDeviceToHost );

    // end time
    auto end = high_resolution_clock::now();

     // the required to call saxpy
    auto duration = duration_cast<microseconds>(end-start);
    std::cout << "time taken by sxy function  " << duration.count() << "  microseconds" << std::endl;
    std::cout << n/duration.count() << "  MFLOPS" << std::endl;

    // write the results of the first and last vector sums
    std::cout << "first element: " << x[0] << "+" << y[0] << "=" << z[0] << std::endl;
    std::cout << "last element: " << x[n-1] << "+" << y[n-1] << "=" << z[n-1] << std::endl;

    // free the memory allocated on host 
    delete [] x;
    delete [] y;
    delete [] z;

    // free the memory allocated on device 
    cudaFree (x);
    cudaFree (y);
    cudaFree (z);

    return 0;
}
