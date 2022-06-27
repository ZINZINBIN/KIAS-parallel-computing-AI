// compile options cc -std=c99 -O3 heat.c 
// elapsed time = 2.200000e+01 seconds with DIM=512
// elapsed time = 3.680000e+01 seconds with DIM=1024

#include <stdio.h>
#include <stdlib.h>  // for malloc()
#include <time.h>    // for the measurement of elapsed time

const int DIM = 512;
const float LMIN = -4.0f;  // computational domain = [LMAX x LMAX]
const float LMAX =  4.0f;  // computational domain = [LMAX x LMAX]
const float TEND = 1.0f;
const float R = 0.25f; // R = kappa*dt/h^2
const float TEMP_MAX =1.0f;
const float TEMP_MIN =0.0001f;

void update_temp ( float * tempIn, float * tempOut ) {

    for (int j=0; j<DIM; j++) {
    for (int i=0; i<DIM; i++) {
        int offset = i + j * DIM;

        int left = offset - 1;
        int right = offset + 1;
        // continuous boundary condition
        if (i == 0) left++;
        if (i == DIM-1) right--;

        int top = offset - DIM;
        int bottom = offset + DIM;
        // continuous boundary condition
        if (j == 0) top += DIM;
        if (j == DIM-1) bottom -= DIM;

        tempOut[offset] = R*(tempIn[left]+tempIn[right]+tempIn[top]+tempIn[bottom])
                         + (1.0f-4.0f*R) * tempIn[offset];
    }
    }

}

void Out2In ( float * tempIn, float * tempOut ) {

    for (int j=0; j<DIM; j++) {
    for (int i=0; i<DIM; i++) {
        int offset = i + j * DIM;
        tempIn[offset] = tempOut[offset];
    }
    }
 
}

int main (void) {

    clock_t etime;  // variables for measuring elapsed time  
    etime = clock();   // start to measure the elasped time

// define arrays for temperature
// temp: working array for temperature
// temp_new: updated temperature
    float * tempIn  = (float *) malloc(DIM*DIM * sizeof(float)); 
    float * tempOut = (float *) malloc(DIM*DIM * sizeof(float)); 

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

// time step
    float t = 0.0f;
    float delt = R*h*h;   // kappa=1.0
    while (t < TEND) {
        t = t + delt;
//        printf("t= %f\n", t);
        update_temp ( tempIn, tempOut ); 
        Out2In ( tempIn, tempOut ); 
    }

    etime = (clock()-etime)/CLOCKS_PER_SEC;
    printf("elapsed time = %e seconds\n",(double) etime);

    FILE *fp;
    fp = fopen("temp.dat", "w");
    fwrite(tempIn, sizeof (float), DIM*DIM, fp);
    fclose(fp);

    for (int i=0; i<DIM; i++) {
        int offset = i + DIM/2 * DIM; 
        printf("%e\n", tempIn[offset]);
    }

    free(tempIn);
    free(tempOut);
    free(x); 
    free(y); 

    return 0;
}
