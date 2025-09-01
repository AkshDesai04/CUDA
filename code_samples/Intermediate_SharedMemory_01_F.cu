/*
Aim: Implement naive matrix-vector multiplication, where each thread block computes a portion of the output vector.

Thinking:
1. Naive matrix-vector multiplication requires computing y[i] = sum_j A[i][j] * x[j] for each row i.
2. For CUDA, we assign one thread per output element (row). A thread block will therefore handle a contiguous segment of rows.
3. Each thread calculates the dot product for its assigned row by looping over all columns.
4. No shared memory or tiling is used; this is the simplest (naive) implementation.
5. Kernel launch parameters:
   - blockDim.x = a reasonable number of threads per block (e.g., 256).
   - gridDim.x = ceil(M / blockDim.x) to cover all rows.
6. Host code:
   - Allocate host arrays A (M x N), x (N), y (M).
   - Initialize A and x with example values (e.g., random or simple pattern).
   - Allocate device memory for A_d, x_d, y_d.
   - Copy A and x to device.
   - Launch kernel.
   - Copy result y back to host.
   - Optionally verify correctness by computing reference result on CPU.
   - Clean up resources.
7. Include error-checking macro for CUDA calls.
8. Provide timing using cudaEvent_t for performance measurement (optional but useful).
9. Ensure the program compiles as a .cu file and can be built with nvcc.
10. The entire code is self-contained and does not require external files.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel: each thread computes one element of the output vector y */
__global__ void matVecMul(const float *A, const float *x, float *y, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}

/* CPU reference implementation for verification */
void matVecMulCPU(const float *A, const float *x, float *y, int M, int N)
{
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

/* Utility to fill array with random floats */
void fillRandom(float *arr, int size)
{
    for (int i = 0; i < size; ++i) {
        arr[i] = (float)rand() / (float)(RAND_MAX);
    }
}

int main(int argc, char *argv[])
{
    /* Matrix dimensions */
    int M = 1024;  // number of rows
    int N = 1024;  // number of columns

    /* Allow override via command line */
    if (argc >= 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }

    printf("Matrix-Vector multiplication: M = %d, N = %d\n", M, N);

    /* Seed random number generator */
    srand((unsigned)time(NULL));

    /* Host memory allocation */
    size_t size_A = M * N * sizeof(float);
    size_t size_x = N * sizeof(float);
    size_t size_y = M * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_x = (float *)malloc(size_x);
    float *h_y = (float *)malloc(size_y);
    float *h_y_ref = (float *)malloc(size_y);

    if (!h_A || !h_x || !h_y || !h_y_ref) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host data */
    fillRandom(h_A, M * N);
    fillRandom(h_x, N);
    /* Zero output arrays */
    memset(h_y, 0, size_y);
    memset(h_y_ref, 0, size_y);

    /* Device memory allocation */
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void **)&d_x, size_x));
    CUDA_CHECK(cudaMalloc((void **)&d_y, size_y));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int BLOCK_SIZE = 256;
    int gridSize = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Record start event */
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* Launch kernel */
    matVecMul<<<gridSize, BLOCK_SIZE>>>(d_A, d_x, d_y, M, N);

    /* Record stop event */
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost));

    /* Compute reference result on CPU */
    matVecMulCPU(h_A, h_x, h_y_ref, M, N);

    /* Verify correctness */
    int errors = 0;
    for (int i = 0; i < M; ++i) {
        float diff = fabs(h_y[i] - h_y_ref[i]);
        if (diff > 1e-3f) {
            errors++;
            if (errors <= 10) {
                printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_y[i], h_y_ref[i]);
            }
        }
    }
    if (errors == 0) {
        printf("Result verified: no errors.\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    /* Clean up */
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
