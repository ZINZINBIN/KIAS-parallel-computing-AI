// written by Jongsoo Kim
// last modification: 2011-06-28
// JK added openACC compiler directives
// compilation

#include <stdio.h>
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 4096;

double dot_product(double *restrict a, double *restrict b) { 

    double c = 0.0;
    for (int i=0; i<N; i++)
        c += a[i]*b[i];
    return c;
}

int main (void) {

    double a[N], b[N];

    for (int i=0; i<N; i++) {
       a[i] = (double) i;
       b[i] = (double) i;
    }

    double c = dot_product(a,b);

    printf("Dot prodoct of a and b = %f\n", c);
    printf("sum_squares of (N-1) = %f\n", sum_squares((double)(N-1)) );

    return 0;
}
