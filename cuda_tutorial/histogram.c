// last modification: 2011-06-28
// compile options cc -std=c99 histogram.c 

#include <stdio.h>
#include <stdlib.h> // rand() function
#include <time.h>   // for the clock() function

const long SIZE = 100*1024*1024; 

int main (void) {

    unsigned char buffer[SIZE];
    unsigned int histo[256];

    for (int i=0; i<SIZE; i++)
        buffer[i] = rand()%256;
        
    double etime = (double) clock(); // start to measure the elapsed time

    for (int i=0; i<256; i++)
        histo[i] = 0;

    for (int i=0; i<SIZE; i++)
        histo[buffer[i]]++;

    etime = ((double) clock() - etime)/(double)CLOCKS_PER_SEC ;
    printf("elapsed time in seconds = %f\n",etime);

    long histoCount = 0;
    for (int i=0; i<256; i++)
        histoCount += histo[i];

    printf ("SIZE: %ld\n", SIZE); 
    printf ("Histogram Sum: %ld\n", histoCount); 

    return 0;
}
