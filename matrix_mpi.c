#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 800  // Define the size of the matrices

void print_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void matrix_multiply(int A[N][N], int B[N][N], int C[N][N], int start_row, int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrix_add(int A[N][N], int B[N][N], int C[N][N], int start_row, int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void matrix_subtract(int A[N][N], int B[N][N], int C[N][N], int start_row, int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void matrix_transpose(int A[N][N], int C[N][N], int start_row, int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            C[j][i] = A[i][j];
        }
    }
}

void matrix_scalar_multiply(int A[N][N], int C[N][N], int scalar, int start_row, int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] * scalar;
        }
    }
}

int matrix_determinant_2x2(int A[N][N]) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

int matrix_determinant_3x3(int A[N][N]) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

int main(int argc, char* argv[]) {
    int rank, size;
    int A[N][N], B[N][N], C_mult[N][N] = {0}, C_add[N][N] = {0}, C_sub[N][N] = {0};
    int C_trans[N][N] = {0}, C_scalar[N][N] = {0};
    int i, j, scalar = 2;
    double start_time, end_time, elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    int remaining_rows = N % size;
    int start_row, end_row;

    // Determine the range of rows each process will handle
    if (rank < remaining_rows) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc + 1;
    } else {
        start_row = rank * rows_per_proc + remaining_rows;
        end_row = start_row + rows_per_proc;
    }

    // Initialize matrices A and B in the root process
    if (rank == 0) {
        printf("Matrix A:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
                printf("%d ", A[i][j]);
            }
            printf("\n");
        }

        printf("Matrix B:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%d ", B[i][j]);
            }
            printf("\n");
        }
    }

    // Start timing
    start_time = MPI_Wtime();

    // Broadcast matrix B to all processes
    MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of matrix A to all processes
    if (rank == 0) {
        // Distribute the rows manually
        for (int p = 1; p < size; p++) {
            int p_start_row, p_end_row;
            if (p < remaining_rows) {
                p_start_row = p * (rows_per_proc + 1);
                p_end_row = p_start_row + rows_per_proc + 1;
            } else {
                p_start_row = p * rows_per_proc + remaining_rows;
                p_end_row = p_start_row + rows_per_proc;
            }
            MPI_Send(&A[p_start_row][0], (p_end_row - p_start_row) * N, MPI_INT, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&A[start_row][0], (end_row - start_row) * N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Perform matrix multiplication on the assigned rows
    matrix_multiply(A, B, C_mult, start_row, end_row);
    // Perform matrix addition on the assigned rows
    matrix_add(A, B, C_add, start_row, end_row);
    // Perform matrix subtraction on the assigned rows
    matrix_subtract(A, B, C_sub, start_row, end_row);
    // Perform matrix transpose on the assigned rows
    matrix_transpose(A, C_trans, start_row, end_row);
    // Perform matrix scalar multiplication on the assigned rows
    matrix_scalar_multiply(A, C_scalar, scalar, start_row, end_row);

    // Gather the results from all processes for multiplication
    if (rank == 0) {
        // Collect the results manually for multiplication
        for (int p = 1; p < size; p++) {
            int p_start_row, p_end_row;
            if (p < remaining_rows) {
                p_start_row = p * (rows_per_proc + 1);
                p_end_row = p_start_row + rows_per_proc + 1;
            } else {
                p_start_row = p * rows_per_proc + remaining_rows;
                p_end_row = p_start_row + rows_per_proc;
            }
            MPI_Recv(&C_mult[p_start_row][0], (p_end_row - p_start_row) * N, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&C_add[p_start_row][0], (p_end_row - p_start_row) * N, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&C_sub[p_start_row][0], (p_end_row - p_start_row) * N, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&C_trans[0][p_start_row], (p_end_row - p_start_row) * N, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&C_scalar[p_start_row][0], (p_end_row - p_start_row) * N, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(&C_mult[start_row][0], (end_row - start_row) * N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&C_add[start_row][0], (end_row - start_row) * N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&C_sub[start_row][0], (end_row - start_row) * N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&C_trans[0][start_row], (end_row - start_row) * N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&C_scalar[start_row][0], (end_row - start_row) * N, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Stop timing
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    // Print the result in the root process
    if (rank == 0) {
        printf("Resulting Matrix C (Multiplication):\n");
        print_matrix(C_mult);
        printf("Resulting Matrix C (Addition):\n");
        print_matrix(C_add);
        printf("Resulting Matrix C (Subtraction):\n");
        print_matrix(C_sub);
        printf("Resulting Matrix C (Transpose):\n");
        print_matrix(C_trans);
        printf("Resulting Matrix C (Scalar Multiplication):\n");
        print_matrix(C_scalar);

        // Compute determinant for small matrices
        if (N == 2) {
            printf("Determinant of Matrix A: %d\n", matrix_determinant_2x2(A));
        } else if (N == 3) {
            printf("Determinant of Matrix A: %d\n", matrix_determinant_3x3(A));
        }
        else {
            printf("Matrix is too large\n");
        }

        // Print time
        printf("Elapsed time: %f seconds\n", elapsed_time);
    }

    MPI_Finalize();
    return 0;
}
