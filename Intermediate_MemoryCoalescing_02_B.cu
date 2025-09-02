/*
Phase 2: __syncthreads() to ensure the whole tile is loaded.

Thinking:
- The goal is to demonstrate the use of __syncthreads() in a tiled matrix multiplication kernel.
- Two matrices A and B are multiplied to produce C. The matrices are square and their size is a multiple of the tile dimension.
- Each block loads a TILE_SIZE x TILE_SIZE tile of A and B into shared memory. The __syncthreads() call is placed immediately after the tile has been fully loaded and before it is used for multiplication, ensuring that all threads have finished copying data into shared memory.
- After the multiplication for the current tile, another __syncthreads() is used before loading the next tile to avoid race conditions.
- The host code allocates memory, initializes matrices with random values, copies them to the device, launches the kernel, copies the result back, and optionally verifies correctness.
- Basic error checking is performed after CUDA API calls.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16   // You can adjust this value

// Kernel for tiled matrix multiplication
__global__ void matrixMulShared(const float *A, const float *B, float *C, int N) {
    // Allocate shared memory for tiles of A and B
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index of C element to work on
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles of input matrices
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A into shared memory
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (aRow < N && aCol < N)
            sA[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < N && bCol < N)
            sB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// Helper function for error checking
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int N = 1024; // Matrix dimension (must be multiple of TILE_SIZE for simplicity)
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize matrices with random values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);
    checkCudaError("cudaMalloc");

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy H2D");

    // Set up execution configuration
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                       (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError("Kernel launch");

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy D2H");

    // Simple verification (optional)
    // Compute one element on CPU and compare
    int testRow = 0, testCol = 0;
    float cpuSum = 0.0f;
    for (int k = 0; k < N; ++k) {
        cpuSum += h_A[testRow * N + k] * h_B[k * N + testCol];
    }
    printf("GPU result at (0,0): %f\n", h_C[testRow * N + testCol]);
    printf("CPU reference at (0,0): %f\n", cpuSum);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
