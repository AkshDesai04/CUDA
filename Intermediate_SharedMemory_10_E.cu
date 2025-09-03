/*
Aim of the program: Explain why reading s_tile[threadIdx.y][threadIdx.x] is fine but writing s_tile[threadIdx.y][threadIdx.x] could cause bank conflicts if threadIdx.y is the faster-changing index.

The program demonstrates how a 2‑D shared memory tile is accessed by threads in a block.  
When the outer index (threadIdx.y) changes the fastest (i.e., each row of the tile is
filled by consecutive threads), the accesses are coalesced in shared memory and do
not conflict.  However, if threads write to the same row of the tile (i.e., the
fastest-changing index is threadIdx.x), several threads may target the same shared
memory bank simultaneously, resulting in bank conflicts.  The code below illustrates
this behavior by performing a simple matrix multiplication that uses a shared
memory tile.  The comments highlight the critical access patterns and explain why
reading is safe while writing may introduce conflicts.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 32

__global__ void matrixMulShared(float *C, const float *A, const float *B,
                                int width, int height, int depth)
{
    // Shared memory tile
    __shared__ float As[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts on writes
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1];

    int bx = blockIdx.x;   // block column
    int by = blockIdx.y;   // block row
    int tx = threadIdx.x;  // thread column
    int ty = threadIdx.y;  // thread row

    // Identify the global row and column of C that this thread will compute
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float Cvalue = 0.0f;

    // Loop over the tiles of A and B that are required to compute C[row][col]
    for (int m = 0; m < (depth + TILE_DIM - 1) / TILE_DIM; ++m) {
        // Load tiles from global memory to shared memory
        // Reading from global memory into shared memory
        // Each thread loads one element into As and Bs
        if (row < height && (m * TILE_DIM + tx) < depth)
            As[ty][tx] = A[row * depth + m * TILE_DIM + tx];
        else
            As[ty][tx] = 0.0f;

        if ((m * TILE_DIM + ty) < depth && col < width)
            Bs[ty][tx] = B[(m * TILE_DIM + ty) * width + col];
        else
            Bs[ty][tx] = 0.0f;

        // Synchronize to ensure the tile is fully loaded before computing
        __syncthreads();

        // Compute the partial product for this tile
        for (int k = 0; k < TILE_DIM; ++k)
            Cvalue += As[ty][k] * Bs[k][tx];

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (row < height && col < width)
        C[row * width + col] = Cvalue;
}

/*
Explanation of bank conflicts:

Shared memory in CUDA is divided into 32 banks.  Each bank can service one
access per cycle if the accesses are to different addresses.  When multiple
threads access the same bank simultaneously, a conflict occurs and the
accesses are serialized, hurting performance.

Reading from As[ty][tx] or Bs[ty][tx]:
  - Here, ty is the outer index and changes slowly across threads.
  - Each thread in a warp accesses a different bank because the row
    (ty) is the same for all threads in a warp, but the column (tx) is
    different.  The layout As[ty][tx] places consecutive columns in
    consecutive banks, so each thread gets a unique bank.  This access
    pattern is conflict‑free.

Writing to As[ty][tx] (or Bs[ty][tx]) when ty is the faster-changing
index:
  - If the threads were reordered so that ty changes faster than tx
    (e.g., if we used As[tx][ty] instead), many threads would write to
    the same bank because consecutive threads in a warp would share the
    same row index (tx).  This would cause a bank conflict.
  - Even with the current layout, if we pad the second dimension
    (As[ty][tx + 1]) we can avoid write conflicts because the padding
    ensures that writes are distributed across banks.  The code above
    uses a +1 padding for this reason.

In summary, reading from a 2‑D tile where the outer index is the slower
changing dimension is safe from bank conflicts.  Writing to the same
layout can cause conflicts if the faster-changing dimension is the
row index; padding or reorganizing the tile can mitigate this.
*/

int main(void)
{
    const int width = 1024;
    const int height = 1024;
    const int depth = 1024;
    size_t sizeA = width * depth * sizeof(float);
    size_t sizeB = depth * height * sizeof(float);
    size_t sizeC = width * height * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize matrices with some values
    for (int i = 0; i < width * depth; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < depth * height; ++i) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    // Launch kernel
    matrixMulShared<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, width, height, depth);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Simple check
    printf("C[0] = %f\n", h_C[0]);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
