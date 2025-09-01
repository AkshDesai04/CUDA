/*
Aim: Declare the shared memory tiles using `__shared__ float a_tile[TILE_SIZE][TILE_SIZE];` syntax.

Thinking:
The task is to create a complete CUDA program (.cu file) that demonstrates how to declare and use shared memory tiles
with the specific syntax provided. To make the example concrete, a simple tiled matrix multiplication kernel is implemented.
The kernel uses two shared memory tiles, `a_tile` and `b_tile`, each declared with the requested syntax:
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
The tile size is chosen as a compile‑time constant (TILE_SIZE = 16) which works well for many GPUs.
Boundary checks are performed to handle non‑multiples of TILE_SIZE. The host code allocates memory for two input matrices
(A and B) and a result matrix (C), initializes them, copies them to the device, launches the kernel, copies back the
result, and prints a few elements for verification. Error checking macros are included for robustness.
The program is fully self‑contained and can be compiled with `nvcc matrix_tiled.cu -o matrix_tiled` and run on a CUDA
capable device.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define CHECK_CUDA(call)                                                     \
    do {                                                                      \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

// Tiled matrix multiplication kernel
__global__ void matMulTiled(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Declare shared memory tiles
        __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
        __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

        // Load tile of A into shared memory
        int tiledCol = m * TILE_SIZE + threadIdx.x;
        if (row < N && tiledCol < N) {
            a_tile[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        } else {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int tiledRow = m * TILE_SIZE + threadIdx.y;
        if (tiledRow < N && col < N) {
            b_tile[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        } else {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    const int N = 1024; // Matrix dimension
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand()) / RAND_MAX;
        h_B[i] = (float)(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print a few results for verification
    printf("C[0] = %f\n", h_C[0]);
    printf("C[%d] = %f\n", N*N/2, h_C[N*N/2]);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
