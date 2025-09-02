/*
Aim of the program: Use padding in the shared memory array (`__shared__ float s_tile[TILE_SIZE][TILE_SIZE+1]`) to eliminate bank conflicts even with non-optimal access patterns.

My thinking:

- The main challenge is to demonstrate how padding a shared memory array eliminates bank conflicts that arise from certain access patterns. 
- I chose a simple and familiar problem: matrix multiplication (C = A * B). 
- Each thread block will compute a TILE_SIZE x TILE_SIZE tile of the result matrix. 
- Two tiles (one from A and one from B) are loaded into shared memory. 
- The shared memory arrays are declared as 2D arrays of size [TILE_SIZE][TILE_SIZE+1]. The +1 introduces padding so that consecutive rows are offset by one element, preventing multiple threads from accessing the same memory bank when they read the shared memory. 
- The access pattern in the kernel (As[threadIdx.y][k] * Bs[k][threadIdx.x]) is deliberately written in a way that would cause bank conflicts if the shared memory were not padded. 
- By padding, each element in a column is placed in a different bank, thus eliminating conflicts. 
- I also added a simple error‑checking macro and a small host routine that initializes matrices, runs the kernel, and verifies the result. 
- The program is fully self‑contained and should compile with nvcc and run on a CUDA capable GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Kernel performing matrix multiplication using padded shared memory
__global__ void matMulShared(const float *A, const float *B, float *C, int N)
{
    // Shared memory tile for A with padding (+1 column)
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    // Shared memory tile for B with padding (+1 column)
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Global row and column indices for this thread
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    // Number of tiles to iterate over
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t)
    {
        // Load tile of A into shared memory
        int tiledColA = t * TILE_SIZE + threadIdx.x;
        if (row < N && tiledColA < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + tiledColA];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        int tiledRowB = t * TILE_SIZE + threadIdx.y;
        if (tiledRowB < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write the computed value to global memory
    if (row < N && col < N)
        C[row * N + col] = Cvalue;
}

// CPU reference implementation for verification
void matMulCPU(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// Main host function
int main(void)
{
    const int N = 512; // Size of matrices (NxN)
    const size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes);

    // Initialize matrices with some values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matMulShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify result with CPU reference
    matMulCPU(h_A, h_B, h_C_ref, N);

    // Simple error check
    double maxError = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double err = fabs(h_C[i] - h_C_ref[i]);
        if (err > maxError) maxError = err;
    }

    printf("Maximum error: %e\n", maxError);

    // Free resources
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
