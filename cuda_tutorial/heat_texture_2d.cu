// written by Jongsoo Kim
// Last modification: 2014-06-09
// compile options nvcc -O3 heat_texture_2d.cu 
// serial: elapsed time  2.200000e+01 seconds for DIM=512
// global memory: elapsed time  1.172689 seconds for DIM=512
// texture 1D: elapsed time: 0.700451 seconds for DIM=512
// texture 2D: elapsed time: 0.700678 seconds for DIM=512

#include <stdio.h>
#include <stdlib.h>  // for malloc()
#include <time.h>    // for the measurement of elapsed time

// there is no way to pass the input and output buffers as parameters.
// texture reference must be declared globally at file scope.
texture<float,2,cudaReadModeElementType> texIn;
texture<float,2,cudaReadModeElementType> texOut;

const int DIM = 512;
const float LMIN = -4.0f;  // computational domain = [LMAX x LMAX]
const float LMAX =  4.0f;  // computational domain = [LMAX x LMAX]
const float TEND = 1.0f;
const float R = 0.25f; // R = dt/h^2
const float TEMP_MAX =1.0f;
const float TEMP_MIN =0.0001f;

__global__ void update_temp ( float * temp, bool InOut ) {

    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = i + j * blockDim.x * gridDim.x;

// no longer need linear mappings
// continuous boundary condition is provided by tex2D as a default

    float t, l, c, r, b;

// since we alternatively update from texIn to texOut and
// from texOut ot texIn, we need not swap function
    if (InOut) {
        t = tex2D(texIn,i,j-1);
        l = tex2D(texIn,i-1,j);
        c = tex2D(texIn,i,j);
        r = tex2D(texIn,i+1,j);
        b = tex2D(texIn,i,j+1);
    }
    else {
        t = tex2D(texOut,i,j-1);
        l = tex2D(texOut,i-1,j);
        c = tex2D(texOut,i,j);
        r = tex2D(texOut,i+1,j);
        b = tex2D(texOut,i,j+1);
    }

    temp[offset] = R*(l+r+t+b) + (1.0f-4.0f*R) * c;

}

int main (void) {

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start,0 );

// define arrays for temperature
// temp: working array for temperature
// temp_new: updated temperature
    float * tempIn, * tempOut; 
    tempIn  = (float *) malloc(DIM*DIM * sizeof(float));
    tempOut = (float *) malloc(DIM*DIM * sizeof(float));

// define device arrays
    float * dev_tempIn, * dev_tempOut;
    cudaMalloc( (void**)&dev_tempIn,  DIM*DIM * sizeof(float) );
    cudaMalloc( (void**)&dev_tempOut, DIM*DIM * sizeof(float) );

// bind device allocations to texture references
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D( NULL, texIn,  dev_tempIn,  desc, DIM, DIM, DIM * sizeof(float) );
    cudaBindTexture2D( NULL, texOut, dev_tempOut, desc, DIM, DIM, DIM * sizeof(float) );

// size of each cell
    float h = (LMAX-LMIN)/DIM;

// a square computational domain 
    float * x = (float *) malloc (DIM * sizeof(float));
    float * y = (float *) malloc (DIM * sizeof(float));

    for (int i=0; i<DIM; i++) {
        x[i] = LMIN + h * (i+0.5f);
        y[i] = x[i]; 
    }

// initial temperature distribution 
    for (int j=0; j<DIM; j++) {
    for (int i=0; i<DIM; i++) {
        int offset = i + j * DIM;
        if (x[i]*x[i]+y[j]*y[j] < 1.0)
           tempIn[offset] = TEMP_MAX; 
        else
           tempIn[offset] = TEMP_MIN; 
    }
    }

    cudaMemcpy ( dev_tempIn, tempIn, DIM*DIM * sizeof(float), cudaMemcpyHostToDevice );

    dim3 blocks(DIM/16,DIM/16,1);
    dim3 threads(16,16,1);

// time step
    float t = 0.0f;
    float delt = R*h*h;
    volatile bool InOut = "true";

    while (t < TEND) {

        t = t + delt;

        float *temp;
        if (InOut) temp = dev_tempOut;
        else temp = dev_tempIn;
        
        update_temp<<<blocks,threads>>>( temp, InOut ); 
        InOut = !InOut;
    }

// Depending on output value of dstOut, the most updated temperature distribution
// is either in tempIn or tempOut. 
    if (!InOut)  cudaMemcpy ( tempIn, dev_tempOut, DIM*DIM * sizeof(float), cudaMemcpyDeviceToHost );
    else cudaMemcpy ( tempIn, dev_tempIn, DIM*DIM * sizeof(float), cudaMemcpyDeviceToHost );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    printf("elapsed time: %f seconds\n", elapsedTime/1000.0 );

    FILE *fp;
    fp = fopen("temp.dat", "w");
    fwrite(tempIn, sizeof (float), DIM*DIM, fp);
    fclose(fp);

//check tempature across a center
//    for (int i=0; i<DIM; i++) {
//        int offset = i + DIM/2 * DIM;
//        printf("%e\n", tempIn[offset]);
//    }

    cudaUnbindTexture(texIn) ;
    cudaUnbindTexture(texOut) ;

    free(tempIn);
    free(tempOut);
    free(x); 
    free(y); 

    cudaFree(dev_tempIn);
    cudaFree(dev_tempOut);

    return 0;
}
