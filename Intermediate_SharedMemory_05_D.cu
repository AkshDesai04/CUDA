```cuda
/*
Aim of the program:
Implement this for rectangular matrices as well (e.g. C(M,N) = A(M,K) * B(K,N)).

Thinking and design:
- The goal is to multiply two matrices A (size M x K) and B (size K x N) on the GPU.
- We use a straightforward CUDA kernel where each thread computes one element of the
  output matrix C (size M x N).  The thread indices map to the (row, col) of C.
- Because the matrices can be rectangular, the kernel must guard against threads
  that fall outside the bounds of the matrices.
- We choose a block size of 16x16 for good occupancy on most GPUs; this can be tuned
  if needed.  The grid dimensions are calculated so that all M rows and N columns
  are covered.
- Host code allocates host and device memory, initializes A and B with random
  double values, copies them to the device, launches the kernel, copies the result
  back, and finally frees memory.  A simple check prints the first few elements
  of the result.
- CUDA error checking is performed after each API call to catch any issues early.
- Simple timing using cudaEvent_t is included to show how long the GPU kernel
  takes.

The program can be compiled with:
    nvcc -o matmul_rect matmul_rect.cu
and run with:
    ./matmul_rect

The matrix dimensions are defined at compile time via the constants M, K, N.
Feel free to change them to test different sizes.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Matrix dimensions (can be changed)
#define M 512     // Number of rows in A and C
#define K 256     // Number of columns in A and rows in B
#define N 384     // Number of columns in B and C

// CUDA kernel for matrix multiplication
__global__ void matMulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C

    if (row < M && col < N)
    {
        double sum = 0.0;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Utility function to allocate and initialize a matrix with random doubles
void initMatrix(double* mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        mat[i] = ((double)rand() / RAND_MAX) * 10.0; // Random values 0-10
    }
}

// Main function
int main(int argc, char* argv[])
{
    // Seed random number generator
    srand(0);

    size_t sizeA = M * K * sizeof(double);
    size_t sizeB = K * N * sizeof(double);
    size_t sizeC = M * N * sizeof(double);

    // Allocate host memory
    double *h_A = (double*)malloc(sizeA);
    double *h_B = (double*)malloc(sizeB);
    double *h_C = (double*)malloc(sizeC);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    // h_C will store the result

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    // Copy host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch kernel
    CUDA_CHECK(cudaEventRecord(start));
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate elapsed time
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Print first 10 elements of result matrix as a sanity check
    printf("First 10 elements of result matrix C:\n");
    for (int i = 0; i < 10 && i < M * N; ++i)
    {
        printf("%0.4f ", h_C[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```