/*
Use a tile size of 16x16, meaning your thread block will also be 16x16.

[Thinking: 
We are creating a CUDA C program that performs tiled matrix multiplication using a 16x16 tile. 
The kernel will load submatrices (tiles) of A and B into shared memory, compute partial dot products,
and accumulate into the result matrix C. 
The tile size is fixed to 16, matching the thread block dimensions of 16x16.
The program sets up host matrices, transfers them to the device, launches the kernel,
and then transfers the result back for verification. 
Basic CUDA error checking is included, and timing can be measured with CUDA events if desired. 
The code is self‑contained and can be compiled with nvcc as a .cu file. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define N 1024  // Matrix dimension (must be multiple of TILE_SIZE for this example)

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",       \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),            \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel: tiled matrix multiplication
__global__ void matMulTiled(const float *A, const float *B, float *C, int n)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index of C to compute
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index of C to compute

    float sum = 0.0f;

    // Loop over tiles of A and B required to compute C[row][col]
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;  // Should be n/TILE_SIZE if divisible

    for (int t = 0; t < numTiles; ++t)
    {
        // Load tile from A
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (aRow < n && aCol < n)
            As[threadIdx.y][threadIdx.x] = A[aRow * n + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < n && bCol < n)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * n + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write the result to C
    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main(void)
{
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host matrices with simple values
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            h_A[i * N + j] = 1.0f;  // You can use random or any pattern
            h_B[i * N + j] = 2.0f;
        }
    }

    // Device pointers
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result (for a few elements)
    int errors = 0;
    for (int i = 0; i < N && errors < 10; ++i)
    {
        for (int j = 0; j < N && errors < 10; ++j)
        {
            float expected = 0.0f;
            for (int k = 0; k < N; ++k)
                expected += h_A[i * N + k] * h_B[k * N + j];
            if (abs(h_C[i * N + j] - expected) > 1e-5f)
            {
                printf("Mismatch at (%d,%d): host=%f, device=%f, expected=%f\n",
                       i, j, h_C[i * N + j], expected, h_C[i * N + j]);
                errors++;
            }
        }
    }

    if (errors == 0)
        printf("Matrix multiplication succeeded. No errors detected.\n");
    else
        printf("Matrix multiplication had %d errors.\n", errors);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
