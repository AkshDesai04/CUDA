/*
Aim: Write the loading phase: each thread in the block loads one element of A's tile and one element of B's tile into shared memory.

Thinking process:
1. The loading phase is the part of a tiled matrix multiplication kernel where each thread copies a single element of the sub‑matrix (tile) of A and B into shared memory so that the rest of the kernel can use these fast accesses.
2. We will use a fixed tile size (e.g., 16×16) and declare two shared memory arrays `sharedA` and `sharedB`. Padding (e.g., `BLOCK_SIZE+1`) is added to reduce bank conflicts.
3. Inside the kernel, each thread calculates its global row and column, then uses its local thread indices (`tx`, `ty`) to load `A[globalRow + k*BLOCK_SIZE]` and `B[globalCol + k*BLOCK_SIZE]` into `sharedA[ty][tx]` and `sharedB[ty][tx]` respectively for each iteration `k` over the tiles.
4. We include a simple `main` that allocates host matrices, copies them to the device, launches the kernel, and copies the result back for verification.
5. The focus here is on the loading phase; other parts (accumulation, boundary handling) are minimal but present for completeness.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Kernel that loads tiles of A and B into shared memory
__global__ void matMulTileKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K)
{
    // Shared memory for tiles of A and B
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflict
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Thread row and column within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices of the element computed by this thread
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float value = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t)
    {
        // Each thread loads one element of A and B into shared memory
        int aRow = row;
        int aCol = t * BLOCK_SIZE + tx;
        if (aRow < M && aCol < K)
            sharedA[ty][tx] = A[aRow * K + aCol];
        else
            sharedA[ty][tx] = 0.0f;

        int bRow = t * BLOCK_SIZE + ty;
        int bCol = col;
        if (bRow < K && bCol < N)
            sharedB[ty][tx] = B[bRow * N + bCol];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads(); // Ensure all data is loaded

        // Compute partial product for this tile
        for (int i = 0; i < BLOCK_SIZE; ++i)
            value += sharedA[ty][i] * sharedB[i][tx];

        __syncthreads(); // Prepare for next tile
    }

    // Write result to global memory
    if (row < M && col < N)
        C[row * N + col] = value;
}

int main()
{
    const int M = 64, K = 128, N = 32;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize host matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(i % 100);
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>((i * 2) % 100);
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matMulTileKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Simple check: print a few elements
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
