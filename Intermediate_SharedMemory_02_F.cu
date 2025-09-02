```cuda
/*
Aim of the program: Explain in comments how using shared memory reduces the number of global memory reads.

Thought process:
1. In CUDA, the memory hierarchy is critical for performance. Global memory is large but slow and has high latency. Shared memory, on the other hand, is small, on‑chip, and has much lower latency. 
2. When many threads in a thread block need the same data, loading that data from global memory once into shared memory allows all the threads to reuse it. This reduces the total number of global memory accesses dramatically.
3. The classic example is matrix multiplication. The naive kernel reads each element of matrix A and B from global memory for every multiplication operation that involves that element. If matrix A has dimension MxK, each element a[i][k] is read K times (once for each j) by threads in the same block. The same goes for B. This results in a huge number of global memory transactions.
4. By using shared memory tiles, each block first loads a sub‑matrix (tile) of A and a sub‑matrix of B into shared memory. Each element of the tile is read from global memory only once per block. Then, within the block, threads use the shared memory copy for all the necessary multiplications. This cuts the number of global memory reads by a factor of the tile dimension.
5. The program below implements a tiled matrix multiplication kernel. The comments inside the kernel explain how shared memory is used and how it reduces global memory reads.

The code demonstrates:
- Allocation of shared memory tiles.
- Cooperative loading of tiles into shared memory.
- Synchronization (__syncthreads) to ensure all threads have the data before computation.
- The final accumulation into the output matrix C.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Tile size for shared memory (must divide M, K, and N for simplicity)
#define TILE_SIZE 16

// CUDA kernel for tiled matrix multiplication
__global__ void matMulShared(const float *A, const float *B, float *C,
                             int M, int N, int K)
{
    // Each thread block computes a TILE_SIZE x TILE_SIZE submatrix of C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // row index of C
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // column index of C

    // Shared memory tiles for A and B
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f; // accumulator for C[row][col]

    // Loop over tiles of A and B that contribute to C[row][col]
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        /* ------------------------------------------------------------------
           1. Load a tile of A into shared memory
           2. Load a tile of B into shared memory
           3. Synchronize to ensure all data is available
           4. Compute partial product for this tile
        ------------------------------------------------------------------- */

        // Global indices for the tile element
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x;
        int Brow = t * TILE_SIZE + threadIdx.y;
        int Bcol = col;

        // Boundary check: if indices exceed matrix dimensions, use 0
        if (Arow < M && Acol < K)
            Asub[threadIdx.y][threadIdx.x] = A[Arow * K + Acol];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if (Brow < K && Bcol < N)
            Bsub[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        // ------------------------------------------------------------------
        // Synchronize to make sure the tile is fully loaded into shared memory
        // This is critical because threads will now read from shared memory.
        // ------------------------------------------------------------------
        __syncthreads();

        // ------------------------------------------------------------------
        // Each thread computes the dot product of the row of Asub
        // and the column of Bsub that it is responsible for.
        // Since Asub and Bsub are in shared memory, this read
        // is much faster than a global memory read.
        // ------------------------------------------------------------------
        for (int i = 0; i < TILE_SIZE; ++i)
            acc += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

        // ------------------------------------------------------------------
        // After the loop over i, we are done with this tile.
        // Threads will load the next tile in the next iteration.
        // Since Asub and Bsub were reused for TILE_SIZE multiplications
        // each element was read from global memory only once per tile,
        // instead of TILE_SIZE times if we had not used shared memory.
        // ------------------------------------------------------------------
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N)
        C[row * N + col] = acc;
}

// Host function to perform matrix multiplication using the above kernel
void runMatMul()
{
    // Matrix dimensions (for demonstration, use small sizes)
    int M = 64; // rows of A and C
    int N = 64; // columns of B and C
    int K = 64; // columns of A and rows of B

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize input matrices with arbitrary values
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Verify result (should be all 2*K = 128)
    for (int i = 0; i < M * N; ++i)
    {
        if (h_C[i] != 2.0f * K)
        {
            printf("Mismatch at index %d: %f\n", i, h_C[i]);
            break;
        }
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    runMatMul();
    return 0;
}
```