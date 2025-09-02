/*
At the end of the kernel, only thread 0 should write the final result from shared_mem[0] to the global memory output.

Thinking process:
1. The user only specified the final write behavior.  
2. To illustrate this correctly, I chose a common parallel reduction pattern that sums an array.  
3. I implemented a two-pass reduction:  
   a. First kernel (`partialSumKernel`) reduces each block into a partial sum stored in `blockSums`.  
   b. Second kernel (`finalSumKernel`) runs with a single block, loads the partial sums into shared memory, reduces them, and writes the final result only from thread 0 to the output pointer.  
4. Host code allocates input data, copies it to the device, launches the kernels, and copies back the final sum.  
5. I added simple error checking macros and used reasonable block and grid sizes.  
6. The code is self‑contained, can be compiled with `nvcc`, and prints the sum of a test array.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                      \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

__global__ void partialSumKernel(const int *input, int *blockSums, int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's partial sum to global memory
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

__global__ void finalSumKernel(const int *blockSums, int *output, int numBlocks)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;

    // Load block sums into shared memory
    sdata[tid] = (tid < numBlocks) ? blockSums[tid] : 0;
    __syncthreads();

    // Reduce across block (single block reduction)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 writes the final result
    if (tid == 0) {
        output[0] = sdata[0];
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_input = (int *)malloc(size);
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1; // simple test: sum should be N
    }

    // Allocate device memory
    int *d_input, *d_blockSums, *d_output;
    CHECK_CUDA(cudaMalloc((void **)&d_input, size));
    // Max number of blocks
    const int BLOCK_SIZE = 256;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMalloc((void **)&d_blockSums, gridSize * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_output, sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // First kernel: compute partial sums
    partialSumKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_input, d_blockSums, N);
    CHECK_CUDA(cudaGetLastError());

    // Second kernel: reduce partial sums to final result
    // Launch with a single block, but size large enough to hold all partial sums
    int finalBlockSize = 1 << 10; // 1024 threads
    int finalGridSize = 1;        // one block
    if (finalBlockSize > gridSize) finalBlockSize = gridSize;
    finalSumKernel<<<finalGridSize, finalBlockSize, finalBlockSize * sizeof(int)>>>(
        d_blockSums, d_output, gridSize);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    int h_output;
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Computed sum = %d, expected = %d\n", h_output, N);

    // Cleanup
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_blockSums));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
