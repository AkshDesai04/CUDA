```cpp
/*
Implement a parallel minimum-finding algorithm.

Thinking:
The goal is to compute the minimum element of an array using CUDA by performing a reduction
across threads in parallel. The typical approach is to have each block load a chunk of the
array into shared memory, perform a reduction within the block to compute a block-local
minimum, and write that minimum to a temporary array. Once all blocks have finished, the
host can perform a final reduction over the block minima, or alternatively a second
kernel launch can reduce those values. For simplicity and to keep the example self‑contained,
the host will perform the final reduction.

Key steps:
1. Allocate host memory and fill with random integers.
2. Allocate device memory and copy the data over.
3. Launch a kernel with enough blocks to cover the array. Each block will:
   - Load its assigned elements into shared memory.
   - Perform a binary tree reduction to find the local minimum.
   - Store the local minimum in a global array of block minima.
4. After the kernel completes, copy the block minima back to the host.
5. Perform a simple linear scan on the host to find the global minimum.
6. Print the result and clean up.

We will use a shared memory buffer sized to the block size and a classic
reduction pattern that works for any block size that is a power of two.
Error checking macros are added for robustness. The example uses integers,
but the same pattern works for floating‑point values as well.

This implementation demonstrates the parallel nature of reduction and
provides a clear, minimal example of a parallel minimum‑finding algorithm
in CUDA C++.*/
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel to compute block-wise minimum
__global__ void reduceMinKernel(const int *input, int *blockMins, size_t N)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread (if within bounds)
    int val = INT_MAX;
    if (idx < N)
        val = input[idx];
    if (idx + blockDim.x < N)
        val = min(val, input[idx + blockDim.x]);

    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0)
    {
        blockMins[blockIdx.x] = sdata[0];
    }
}

int main(void)
{
    const size_t N = 1 << 20;          // Number of elements (1M)
    const size_t bytes = N * sizeof(int);

    // Host allocation
    int *h_data = (int*)malloc(bytes);
    int *h_blockMins = NULL;

    // Seed RNG and fill host array
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i)
    {
        h_data[i] = rand() % 10000;  // Random ints [0,9999]
    }

    // Device allocation
    int *d_data = NULL;
    int *d_blockMins = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));

    // Determine grid and block dimensions
    const unsigned int BLOCK_SIZE = 256;
    unsigned int gridSize = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    size_t blockMinsBytes = gridSize * sizeof(int);
    CHECK_CUDA(cudaMalloc((void**)&d_blockMins, blockMinsBytes));
    h_blockMins = (int*)malloc(blockMinsBytes);

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    reduceMinKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_data, d_blockMins, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy block minima back
    CHECK_CUDA(cudaMemcpy(h_blockMins, d_blockMins, blockMinsBytes, cudaMemcpyDeviceToHost));

    // Final reduction on host
    int globalMin = INT_MAX;
    for (unsigned int i = 0; i < gridSize; ++i)
    {
        if (h_blockMins[i] < globalMin)
            globalMin = h_blockMins[i];
    }

    printf("Global minimum: %d\n", globalMin);

    // Cleanup
    free(h_data);
    free(h_blockMins);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_blockMins));

    return 0;
}
```