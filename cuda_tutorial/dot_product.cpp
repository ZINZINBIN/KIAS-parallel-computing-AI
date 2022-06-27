// written by Jongsoo Kim
// last modification: 2021-07-13
// JK added openACC compiler directives
// compilation
// nvc++ -o dot_product_host -acc=host -fast -Minfo=opt dot_product.cpp 
// nvc++ -o dot_product_gpu -acc=gpu -gpu=cc80 -fast -Minfo=opt dot_product.cpp 

#include <iostream>

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int N = 2048;

void dot_product(const float *restrict a, const float *restrict b, float *c) 
{ 

#pragma acc kernels copyin (a[0:N],b[0:N]) copyout (c)
{
    float tmp = 0.0f;
    #pragma acc loop reduction(+:tmp)
    for (int i=0; i<N; i++)
        tmp += a[i]*b[i];
    *c = tmp;
}
}

int main (void) {

    float a[N], b[N];
    float c;

    for (int i=0; i<N; i++) {
       a[i] = (float) i;
       b[i] = (float) i;
    }

    dot_product(a,b,&c);

    std::cout << "Dot prodoct of a and b = " << c << std::endl;
    std::cout << "sum_squares of (N-1) = " << sum_squares((float)(N-1)) << std::endl;

    return 0;
}
