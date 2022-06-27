// written by Jongsoo Kim
// Last modification: 2014-06-04
// compile options nvcc -O3 heat_global.cu 
// serial: elapsed time  2.200000e+01 seconds for DIM=512
// global memory: elapsed time  1.172689 seconds for DIM=512
// texture 1D: elapsed time: 0.700451 seconds for DIM=512
// texture 2D: elapsed time: 0.700678 seconds for DIM=512

#include <stdio.h>
#include <stdlib.h>  // for malloc()

const int DIM = 512;
const float LMIN = -4.0f;  // computational domain = [LMAX x LMAX]
const float LMAX =  4.0f;  // computational domain = [LMAX x LMAX]
const float TEND = 1.0f;
const float R = 0.25f; // R = dt/h^2
const float TEMP_MAX =1.0f;
const float TEMP_MIN =0.0001f;

__global__ void update_temp ( float * temp_new, float * temp ) {

    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = i + j * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (i == 0) left++;
    if (i == DIM-1) right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (j == 0) top += DIM;
    if (j == DIM-1) bottom -= DIM;

    temp_new[offset] = R*(temp[left]+temp[right]+temp[top]+temp[bottom])
                     + (1.0f-4.0f*R) * temp[offset];

}

__global__ void swap_temp ( float * temp_new, float * temp ) {

    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = i + j * blockDim.x * gridDim.x;

    temp[offset] = temp_new[offset];

}

int main (void) {

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start,0 );

// define arrays for temperature
// temp: working array for temperature
// temp_new: updated temperature
    float * temp, * temp_new; 
    temp     = (float *) malloc(DIM*DIM * sizeof(float));
    temp_new = (float *) malloc(DIM*DIM * sizeof(float));

// define device arrays
    float * dev_temp, * dev_temp_new;
    cudaMalloc( (void**)&dev_temp,     DIM*DIM * sizeof(float) );
    cudaMalloc( (void**)&dev_temp_new, DIM*DIM * sizeof(float) );

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
           temp[offset] = TEMP_MAX; 
        else
           temp[offset] = TEMP_MIN; 
    }
    }

    cudaMemcpy ( dev_temp, temp, DIM*DIM * sizeof(float), cudaMemcpyHostToDevice );

    dim3 blocks(DIM/32,DIM/32,1);
    dim3 threads(32,32,1);

// time step
    float t = 0.0f;
    float delt = R*h*h;

    while (t < TEND) {

        t = t + delt;
//        printf("t= %f\n", t);
        update_temp<<<blocks,threads>>>( dev_temp_new, dev_temp ); 
        swap_temp<<<blocks,threads>>>( dev_temp_new, dev_temp ); 
    }

    cudaMemcpy ( temp, dev_temp, DIM*DIM * sizeof(float), cudaMemcpyDeviceToHost );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    printf("elapsed time: %f seconds\n", elapsedTime/1000.0f );

    FILE *fp;
    fp = fopen("temp.dat", "w");
    fwrite(temp, sizeof (float), DIM*DIM, fp);
    fclose(fp);

//check tempature across a center
//    for (int i=0; i<DIM; i++) {
//        int offset = i + DIM/2 * DIM;
//        printf("%e\n", temp[offset]);
//    }

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    free(temp_new);
    free(temp);
    free(x); 
    free(y); 

    cudaFree( dev_temp_new);
    cudaFree( dev_temp);

    return 0;
}
