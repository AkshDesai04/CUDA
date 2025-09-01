/*
Explain what would happen if you forgot the `__syncthreads()` at the end of the main tile loop. (A race condition where threads start loading the next tile before others have finished computing with the current one).

Thought process:
The goal of this CUDA program is to illustrate the importance of `__syncthreads()` in a tiled matrix multiplication kernel. The program multiplies two square matrices using shared memory tiling. Each thread block loads a tile of the input matrices into shared memory, synchronizes, performs partial multiplication, then proceeds to the next tile. The comment above describes the potential race condition if the final `__syncthreads()` after the inner tile loop is omitted: threads would begin loading the next tile while some threads are still using data from the current tile, leading to incorrect results or undefined behavior. The code below is a complete, self-contained CUDA C example that compiles to a .cu file and demonstrates the correct use of `__syncthreads()`. The main focus is the kernel, but host code is provided to launch the kernel and verify correctness. All extraneous information is omitted as per the developer instructions. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 512          // Matrix size (N x N)
#define TILE_SIZE 32   // Tile dimension (must divide N)

// Kernel for tiled matrix multiplication
__global__ void matMulShared(const float *A, const float *B, float *C, int n)
{
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread row/col within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row/col of the element computed by this thread
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;  // Accumulator for the C element

    // Loop over tiles
    for (int m = 0; m < (n + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load elements of A and B into shared memory
        int aIdx = row * n + (m * TILE_SIZE + tx);
        int bIdx = (m * TILE_SIZE + ty) * n + col;

        As[ty][tx] = (aIdx < n * n) ? A[aIdx] : 0.0f;
        Bs[ty][tx] = (bIdx < n * n) ? B[bIdx] : 0.0f;

        // Wait for all threads to finish loading
        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += As[ty][k] * Bs[k][tx];

        // Wait for all threads to finish computation before next tile
        __syncthreads();
    }

    // Write the result
    if (row < n && col < n)
        C[row * n + col] = acc;
}

// Host function to initialize matrices
void initMatrix(float *M, int n, float val)
{
    for (int i = 0; i < n * n; ++i)
        M[i] = val;
}

// Host function to check result
bool verify(const float *C, int n)
{
    for (int i = 0; i < n * n; ++i) {
        if (abs(C[i] - 2.0f) > 1e-4f) {
            printf("Mismatch at index %d: %f\n", i, C[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    int size = N * N;
    size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize matrices A and B
    initMatrix(h_A, N, 1.0f);
    initMatrix(h_B, N, 1.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matMulShared<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    if (verify(h_C, N))
        printf("Matrix multiplication succeeded.\n");
    else
        printf("Matrix multiplication failed.\n");

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
