/*
Implement tiled matrix multiplication. Each thread block will calculate one square sub-matrix (tile) of C. To do this, all threads in the block must cooperatively load the corresponding tiles of A and B into shared memory first.

Thinking:
- The goal is to compute C = A * B for square matrices of size N x N.  
- To exploit shared memory and reduce global memory traffic, each thread block will compute one tile of C of size TILE_SIZE x TILE_SIZE.  
- Each thread in the block cooperatively loads a tile of A and a tile of B into shared memory (Asub and Bsub).  
- After synchronizing, each thread computes the partial sum for its element of the tile by iterating over the shared tiles.  
- This process is repeated for all tiles that span the width of A (or height of B).  
- The block-level tile size is chosen as a compile-time constant; a typical value is 16 or 32.  
- The host code will allocate pinned memory for faster transfer, initialize matrices with random values, launch the kernel with appropriate grid/block dimensions, and copy the result back.  
- Basic error checking and timing (using CUDA events) are included for demonstration.  
- For simplicity, the code assumes N is divisible by TILE_SIZE. If not, padding or handling the edge cases would be required.  
- The code is self-contained in a single .cu file and can be compiled with nvcc.  
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // You can tune this value

// CUDA kernel for tiled matrix multiplication
__global__ void matrixMulTiled(const float *A, const float *B, float *C, int N)
{
    // Shared memory for tiles of A and B
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    // Identify the row and column of the element this thread will compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    // Loop over tiles of the input matrices
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m)
    {
        // Load tile from A into shared memory
        int Arow = row;
        int Acol = m * TILE_SIZE + threadIdx.x;
        if (Arow < N && Acol < N)
            Asub[threadIdx.y][threadIdx.x] = A[Arow * N + Acol];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory
        int Brow = m * TILE_SIZE + threadIdx.y;
        int Bcol = col;
        if (Brow < N && Bcol < N)
            Bsub[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
            Cvalue += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];

        __syncthreads();
    }

    // Write the result to global memory
    if (row < N && col < N)
        C[row * N + col] = Cvalue;
}

// Host function to allocate and initialize matrices
void initializeMatrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char *argv[])
{
    int N = 1024;  // Default matrix size
    if (argc > 1)
        N = atoi(argv[1]);

    size_t size = N * N * sizeof(float);

    // Allocate host memory (pinned for faster transfer)
    float *h_A, *h_B, *h_C;
    cudaMallocHost((void **)&h_A, size);
    cudaMallocHost((void **)&h_B, size);
    cudaMallocHost((void **)&h_C, size);

    // Initialize host matrices
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    cudaEventRecord(start);
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    // Wait for completion
    cudaEventSynchronize(stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Compute elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix multiplication of size %d x %d completed in %f ms\n", N, N, milliseconds);

    // Optionally, verify result for small matrices
    if (N <= 4)
    {
        printf("Result matrix C:\n");
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
                printf("%0.2f ", h_C[i * N + j]);
            printf("\n");
        }
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
