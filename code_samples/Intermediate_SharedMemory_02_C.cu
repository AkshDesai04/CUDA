```cuda
/*
Aim: Write the loading phase: each thread in the block loads one element of A's tile and one element of B's tile into shared memory.

Thinking:
The requirement is to demonstrate the shared‑memory tiling technique used in matrix multiplication on CUDA.  
For a given block of size BLOCK_SIZE × BLOCK_SIZE, each thread will load a single element from the corresponding
tiles of matrices A and B into shared memory.  After loading, a __syncthreads() call is necessary to ensure all
elements are available before any thread starts computing with them.

The kernel below implements the entire tiled matrix multiplication, but the focus is on the loading phase.
Shared memory arrays `As` and `Bs` hold a tile of A and B respectively.  The loop over `p` iterates over the
number of tiles needed to cover the full width of the input matrices.  Inside the loop each thread loads one
element from global memory into the shared arrays.  After the load we synchronize the threads, perform the
partial product accumulation, and finally write the result to global memory.

The code assumes square matrices of size N×N and that N is a multiple of BLOCK_SIZE for simplicity.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16   // adjust as needed

// Kernel performing tiled matrix multiplication
__global__ void matMulKernel(const float *A, const float *B, float *C, int N)
{
    // Shared memory for tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Identify thread coordinates within the block
    int tx = threadIdx.x;  // column index within block
    int ty = threadIdx.y;  // row index within block

    // Global coordinates of the element to compute
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float Cvalue = 0.0f;

    // Loop over all tiles required for the full dot product
    for (int p = 0; p < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++p)
    {
        // Global indices of the elements to load
        int Arow = row;
        int Acol = p * BLOCK_SIZE + tx;
        int Brows = p * BLOCK_SIZE + ty;
        int Bcol = col;

        // Load one element of A and B into shared memory
        if (Arow < N && Acol < N)
            As[ty][tx] = A[Arow * N + Acol];
        else
            As[ty][tx] = 0.0f;  // handle out-of-bounds

        if (Brows < N && Bcol < N)
            Bs[ty][tx] = B[Brows * N + Bcol];
        else
            Bs[ty][tx] = 0.0f;  // handle out-of-bounds

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Cvalue += As[ty][k] * Bs[k][tx];

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the computed value to global memory
    if (row < N && col < N)
        C[row * N + col] = Cvalue;
}

// Host helper to launch the kernel
void launchMatMul(const float *h_A, const float *h_B, float *h_C, int N)
{
    float *d_A, *d_B, *d_C;
    size_t bytes = N * N * sizeof(float);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Simple test harness
int main()
{
    const int N = 512;  // matrix dimension
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize matrices with some values
    for (int i = 0; i < N * N; ++i)
    {
        h_A[i] = static_cast<float>(i % 100);
        h_B[i] = static_cast<float>((i * 2) % 100);
    }

    launchMatMul(h_A, h_B, h_C, N);

    // Optional: verify a few elements
    for (int i = 0; i < 5; ++i)
        printf("C[%d] = %f\n", i, h_C[i]);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```