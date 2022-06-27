/**
 * @file example02.cpp
 * @author zinzinbin
 * @brief Monte-Carlo Integral code for MPI
 * @version 0.1
 * @date 2022-06-27
 * 
 * How to execute
 * (1) mpic++ example02.cpp -o example02.out
 * (2) mpirun -np <num_procs> example02.out
 */
#include <iostream>
#include <mpi.h>

using namespace std;

double func(double x){

}

int main(int argc, char *argv[])
{

    int rank, size;

    // Initialize
    MPI_Init(&argc, &argv);

    // Initialize the number of procs
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the process number : rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        cout << "rank : " << rank << endl;
    }

    MPI_Finalize();
    return 0;
}