/**
 * @file example02.cpp
 * @author zinzinbin
 * @brief find the prime number for MPI
 * @version 0.1
 * @date 2022-06-28
 *
 * How to execute
 * (1) mpic++ example04.cpp -o example04.out
 * (2) mpirun -np <num_procs> example04.out
 */

#include <iostream>
#include <mpi.h>
#include <math.h>

using namespace std;

int check_prime_num(int m, int n_lower, int n_upper){
    int is_prime = 1;
    for(int l = n_lower; l < n_upper; l ++){
        if(m % l == 0){
            is_prime = 0;
            break;
        }
    }
    return is_prime;
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

    double result = 0;
    double master_buffer = new double[];

    // int n = (int) pow(2, 1024) + 1;
    int n;
    int n_interval = (int) n / 2 / (size - 1);
    int is_prime;

    if (rank == 0)
    {
        MPI_Bcast()

        for(int i = 0; i < size - 1; i ++){
            MPI_Recv();
        }
    }
    else
    {
        int n_lower = rank * n_interval + 1;
        int n_upper = (rank + 1) * n_interval + 1;

        if(rank == size - 1){
            n_upper = n;
        }

        is_prime = check_prime_num(n, n_lower, n_upper);

        MPI_Recv();
        cout << "rank : " << rank << " - process complete" << endl;

        MPI_Send();
    }

    MPI_Finalize();
    return 0;
}