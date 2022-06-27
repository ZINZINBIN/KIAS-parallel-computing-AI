/**
 * @file example01.cpp
 * @author zinzinbin
 * @brief Introduction and Integral example for MPI
 * @version 0.1
 * @date 2022-06-27
 *
 * How to execute
 * (1) mpic++ example01.cpp -o example01.out
 * (2) mpirun -np <num_procs> example01.out
 */

#include <iostream>
#include <mpi.h>

using namespace std;

double function(double x){
    return (double) 4.0 / (1.0 + x*x);
}

double integral(int idx_start, int idx_end, int idx_interval, double x_min, double x_max, int n_interval)
{
    double sum = 0;
    double x;
    double y;
    double dx = (x_max - x_min) / n_interval;
    for(int i = idx_start; i < idx_end; i+=idx_interval){
        x = x_min + i * dx;
        y = function(x);
        sum += y;
    }  
    sum *= dx;
    return sum;
}

int main(int argc, char *argv[]){

    int rank, size;

    // Initialize
    MPI_Init(&argc, &argv);

    // Initialize the number of procs
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the process number : rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0){
        cout << "rank : " << rank << endl;
        cout << "number of processes : " << size << endl;
    }

    // time check
    double start_time;
    double end_time;

    // Integral process
    double xi = 0.0;
    double xf = 1.0;
    double result;
    double result_proc;
    
    int n_interval = 1024 * 16;

    MPI_Bcast(&n_interval, 1, MPI_INT, 0, MPI_COMM_WORLD);

    result_proc = integral(rank, n_interval, size, xi, xf, n_interval);

    MPI_Reduce(&result_proc, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(rank == 0){
        cout << "integral result : " << result << endl;
    }

    MPI_Finalize();
    return 0;
}