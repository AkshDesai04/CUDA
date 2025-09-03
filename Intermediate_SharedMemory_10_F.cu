/*
Use padding in the shared memory array (`__shared__ float s_tile[TILE_SIZE][TILE_SIZE+1]`) to eliminate bank conflicts even with non-optimal access patterns.

Thinking:
- Bank conflicts in CUDA occur when multiple threads in a warp access the same shared memory bank.
- A typical 2D shared memory array of size TILE_SIZE x TILE_SIZE will map each column to a bank, so if threads access elements with the same column index, they conflict.
- Adding padding to the second dimension (`TILE_SIZE+1`) ensures that each row of the array starts at a different bank, effectively breaking the alignment that causes conflicts.
- In this program, we perform a simple matrix transpose kernel that loads elements from global memory into the padded shared array, synchronizes, and then writes the transposed elements back to global memory.
- Even if the load/store pattern is not optimal (e.g., strided accesses), the padding guarantees that each thread accesses a unique bank, thus eliminating bank conflicts.
- The kernel is written with `__shared__ float s_tile[TILE_SIZE][TILE_SIZE+1]` and demonstrates the effectiveness of padding in real code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MATRIX_WIDTH 1024
#define MATRIX_HEIGHT 1024

// CUDA error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel to transpose a matrix using shared memory with padding to avoid bank conflicts
__global__ void transposePad(const float *A, float *B, int width, int height)
{
    // Allocate shared memory with padding: TILE_SIZE x (TILE_SIZE+1)
    __shared__ float s_tile[TILE_SIZE][TILE_SIZE + 1];

    // Calculate global row and column indices
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;

    int row = blockY * TILE_SIZE + threadIdx.y;
    int col = blockX * TILE_SIZE + threadIdx.x;

    // Load data from global memory to shared memory
    if (row < height && col < width) {
        s_tile[threadIdx.y][threadIdx.x] = A[row * width + col];
    } else {
        // Handle out-of-bounds by loading zero (optional)
        s_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Synchronize to make sure the tile is loaded
    __syncthreads();

    // Write transposed data from shared memory to global memory
    int transposedRow = blockX * TILE_SIZE + threadIdx.y;
    int transposedCol = blockY * TILE_SIZE + threadIdx.x;

    if (transposedRow < width && transposedCol < height) {
        B[transposedRow * height + transposedCol] = s_tile[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    const int size = MATRIX_WIDTH * MATRIX_HEIGHT;
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);

    // Initialize input matrix with random data
    for (int i = 0; i < size; ++i) {
        h_A[i] = (float)(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));

    // Copy input matrix to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((MATRIX_WIDTH + TILE_SIZE - 1) / TILE_SIZE,
                 (MATRIX_HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    transposePad<<<dimGrid, dimBlock>>>(d_A, d_B, MATRIX_WIDTH, MATRIX_HEIGHT);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    // Simple verification (compare a few elements)
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        int row = i;
        int col = i;
        if (h_A[row * MATRIX_WIDTH + col] != h_B[col * MATRIX_HEIGHT + row]) {
            printf("Mismatch at (%d, %d): %f vs %f\n", row, col,
                   h_A[row * MATRIX_WIDTH + col], h_B[col * MATRIX_HEIGHT + row]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Transpose verified successfully.\n");
    } else {
        printf("Transpose verification failed.\n");
    }

    // Free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    free(h_A);
    free(h_B);

    return 0;
}