/*
The first kernel will take the large input array and an output array sized to the number of blocks. Each block writes its partial sum to d_partial_sums[blockIdx.x].
I will design a CUDA program that performs a parallel reduction of a large array. The program follows these steps:

1. Host code generates a large input array (e.g., 1 << 20 elements) and allocates device memory for the input, a partial sums array (size equal to the number of blocks), and the final sum.
2. `partial_sum_kernel` runs on the device. Each block reads elements from the input array, performs a block‑level reduction using shared memory, and writes the block’s partial sum to `d_partial_sums[blockIdx.x]`. Threads that read out‑of‑bounds indices contribute zero to the sum.
3. After the first kernel completes, a second kernel, `final_reduce_kernel`, reduces the array of partial sums to a single value. It follows the same block‑level reduction pattern but with only enough blocks to handle the number of partial sums.
4. The host copies the final sum back to host memory, prints it, and frees all allocated resources.
5. The program includes basic error checking for CUDA API calls.

The key points in the kernel logic:
- Use `blockDim.x` threads per block, with `threadIdx.x` indexing into shared memory.
- Load elements from the global array only if the index is less than the total size; otherwise use zero.
- Use a loop that halves the active thread count each iteration, performing pairwise addition.
- After the loop, thread 0 writes the block’s result to the output array.

This implementation demonstrates how to parallelize a reduction using CUDA and how to propagate partial results from each block to compute a global sum.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void partial_sum_kernel(const float *d_in, float *d_partial_sums,
                                   size_t n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    float x = (idx < n) ? d_in[idx] : 0.0f;
    sdata[tid] = x;
    __syncthreads();

    // Block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's partial sum to output array
    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void final_reduce_kernel(float *d_partial_sums, float *d_result,
                                    size_t n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float x = (idx < n) ? d_partial_sums[idx] : 0.0f;
    sdata[tid] = x;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }

    // After the last block, thread 0 writes final result
    if (blockIdx.x == 0 && tid == 0) {
        *d_result = d_partial_sums[0];
    }
}

int main(void)
{
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocation
    float *h_in = (float *)malloc(bytes);
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = 1.0f; // Example: all ones
    }

    // Device allocation
    float *d_in = NULL;
    float *d_partial_sums = NULL;
    float *d_result = NULL;

    CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_partial_sums, N * sizeof(float))); // Max size, will shrink later
    CHECK_CUDA(cudaMalloc((void **)&d_result, sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch partial sum kernel
    partial_sum_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_in, d_partial_sums, N);
    CHECK_CUDA(cudaGetLastError());

    // Now reduce the partial sums
    size_t numPartial = blocksPerGrid;
    while (numPartial > 1) {
        int blocks = (numPartial + threadsPerBlock - 1) / threadsPerBlock;
        final_reduce_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
            d_partial_sums, d_result, numPartial);
        CHECK_CUDA(cudaGetLastError());

        // Prepare for next iteration
        numPartial = blocks;
    }

    // Copy result back to host
    float h_result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Sum = %f\n", h_result);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_partial_sums);
    cudaFree(d_result);
    free(h_in);

    return 0;
}
