// written by Jongsoo Kim
// Last modification: 2014-06-04
// compile options cc vector_sum.c 
// revision history:  
// 20210711 JK added the #progma openACC compiler directive
// 
//   nvc++ -o vector_sum_host -acc=host -fast -Minfo=opt vector_sum.c 
//   nvc++ -o vector_sum_multicore -acc=multicore -fast -Minfo=opt vector_sum.c 
//   nvc++ -o vector_sum_gpu -acc=gpu -gpu=cc70 -fast -Minfo=opt vector_sum.c 


#include <stdio.h>

const int N = 128;

void add(int *a, int *b, int *c) { 

//#pragma acc kernels
    for (int i=0; i<N; i++) {
        c[i] = a[i] + b[i];
    }
}
     
int main (void) {
    int a[N], b[N], c[N];

    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    add (a, b, c);

    for (int i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i],b[i],c[i]);
    }

    return 0;
}
