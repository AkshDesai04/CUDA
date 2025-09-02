```cuda
/* 
Aim of the program:
Use `cudaMallocPitch` and `cudaMemcpy2D` for your matrices and modify the kernel access logic to use the pitch.

Thinking and design:

1. We want a simple, self‑contained CUDA program that demonstrates pitched memory
   allocation for 2‑D arrays (matrices) and the use of `cudaMemcpy2D` for
   transferring data between host and device.

2. Matrix multiplication is a common example that uses 2‑D indexing.  It also
   benefits from shared‑memory tiling, which allows us to show how the pitch
   is handled in a real kernel.

3. Device memory for each matrix (A, B, C) is allocated with
   `cudaMallocPitch`.  The pitch is returned in bytes and may be larger than
   the logical row width to satisfy alignment requirements.

4. The host matrices are stored in contiguous memory (no pitch).  When
   copying to/from the device we specify a source or destination pitch of
   `row * sizeof(float)`.  `cudaMemcpy2D` handles the stride correctly.

5. In the kernel we receive the pitch of each matrix as a `size_t` argument.
   Because the pitch is in bytes, we cast the base pointer to `char*` and
   compute the byte offset manually:
   ```
   const float a = *((const float*)((const char*)A + row * pitchA + k * sizeof(float)));
   ```
   The same technique is used for matrix B.  This keeps the kernel independent
   from any padding.

6. The kernel uses a standard 16×16 tile for shared memory.  Each thread
   loads one element of A and one element of B into shared arrays, then
   performs the partial dot product.  The loop over `k` iterates over tiles of
   the K dimension.

7. After the kernel finishes we copy the resulting matrix C back to the host
   with `cudaMemcpy2D`.

8. Finally we print a small portion of the result to verify correctness
   and free all resources.

This program is written in C/CUDA and can be compiled with `nvcc`:
    nvcc -o pitched_mm pitched_mm.cu
It will perform a single multiplication of two randomly initialized matrices
of size 512×512 and print a few values from the result.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Tile size for shared memory
#define TILE_WIDTH 16

// Kernel for matrix multiplication using pitched memory
__global__ void matMulPitch(const float *A, size_t pitchA,
                            const float *B, size_t pitchB,
                            float *C, size_t pitchC,
                            int M, int N, int K)
{
    // Shared memory for tiles of A and B
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];

    // Global row and column indices for C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles of the K dimension
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Global indices of elements to load
        int tiledRow = row;
        int tiledCol = t * TILE_WIDTH + threadIdx.x; // column index for A
        if (tiledRow < M && tiledCol < K)
        {
            // Load A[tiledRow][tiledCol]
            const char *pA = (const char *)A + tiledRow * pitchA + tiledCol * sizeof(float);
            Asub[threadIdx.y][threadIdx.x] = *((const float *)pA);
        }
        else
        {
            Asub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        tiledRow = t * TILE_WIDTH + threadIdx.y; // row index for B
        tiledCol = col;
        if (tiledRow < K && tiledCol < N)
        {
            // Load B[tiledRow][tiledCol]
            const char *pB = (const char *)B + tiledRow * pitchB + tiledCol * sizeof(float);
            Bsub[threadIdx.y][threadIdx.x] = *((const float *)pB);
        }
        else
        {
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write result to C if within bounds
    if (row < M && col < N)
    {
        float *pC = (float *)((char *)C + row * pitchC + col * sizeof(float));
        *pC = sum;
    }
}

int main()
{
    // Matrix dimensions: A is MxK, B is KxN, C is MxN
    const int M = 512;   // rows of A and C
    const int K = 512;   // columns of A and rows of B
    const int N = 512;   // columns of B and C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory (contiguous)
    float *hA = (float *)malloc(sizeA);
    float *hB = (float *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);

    if (!hA || !hB || !hC) {
        fprintf(stderr, "Failed to allocate host matrices\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host matrices with random data
    srand((unsigned)time(NULL));
    for (int i = 0; i < M * K; ++i) hA[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand() % 100) / 10.0f;

    // Device pitched memory allocation
    float *dA, *dB, *dC;
    size_t pitchA, pitchB, pitchC;

    CHECK_CUDA(cudaMallocPitch((void **)&dA, &pitchA, K * sizeof(float), M));
    CHECK_CUDA(cudaMallocPitch((void **)&dB, &pitchB, N * sizeof(float), K));
    CHECK_CUDA(cudaMallocPitch((void **)&dC, &pitchC, N * sizeof(float), M));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy2D(dA, pitchA, hA, K * sizeof(float),
                            K * sizeof(float), M, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2D(dB, pitchB, hB, N * sizeof(float),
                            N * sizeof(float), K, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    matMulPitch<<<dimGrid, dimBlock>>>(dA, pitchA, dB, pitchB, dC, pitchC,
                                        M, N, K);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy2D(hC, N * sizeof(float), dC, pitchC,
                            N * sizeof(float), M, cudaMemcpyDeviceToHost));

    // Simple validation: print a few elements
    printf("C[0][0] = %f\n", hC[0]);
    printf("C[0][1] = %f\n", hC[1]);
    printf("C[1][0] = %f\n", hC[N]);
    printf("C[1][1] = %f\n", hC[N + 1]);

    // Clean up
    free(hA);
    free(hB);
    free(hC);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}
```