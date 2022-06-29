/**
 * @file example02.cpp
 * @author zinzinbin
 * @brief Matrix Multiplication for MPI
 * @version 0.1
 * @date 2022-06-28
 *
 * How to execute
 * (1) mpic++ example03.cpp -o example03.out
 * (2) mpirun -np <num_procs> example03.out
 */

#include <iostream>
#include <mpi.h>

using namespace std;

float **generate_matrix(int m, int n){
    float mat = new float*[m];
    for(int i = 0; i < m; i++){
        mat[i] = new float[n];  
    }
    return mat
}

float *generate_array(int m){
    float arr = new float [n];
    return arr;
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

    // monte-carlo integral
    if (rank == 0)
    {
        // master process : send random seed and receive each process result
        cout << "rank : 0 - send random seed to each process" << rank << endl;
        MPI_Send();

        MPI_Recv();
    }
    else
    {
        // node : receive random seed and proceed calculation process, then send the result to master process
        MPI_Recv();
        cout << "rank : " << rank << " - receive random seed, calculation process start" << endl;

        // calculation process

        // send result from each node to master node
        MPI_Send();
    }

    MPI_Finalize();
    return 0;
}