```cpp
/* 
Aim of the program: Implement a max-finding reduction using the same pattern.

Thinking:
- The task is to perform a reduction that finds the maximum element in an array using CUDA.
- We’ll follow a classic block‑wise reduction pattern:
  1. Each thread loads one element from global memory into shared memory.
  2. Threads iteratively combine pairs of elements in shared memory, halving the active thread count each step.
  3. After the reduction, the first element of shared memory holds the block's maximum.
  4. The block maximums are stored in a temporary array; a second kernel reduces those to a single global maximum.
- To keep the code simple and self‑contained, the array will be filled with random numbers using the standard C library.
- We’ll use the device function `max` for integer comparison. For double precision we would use `fmax`.
- Edge cases (array size not a multiple of block size) are handled by padding the shared array with a very small value (for max).
- The code is written in CUDA C (file extension .cu) and can be compiled with `nvcc`.
- Comments in the code explain each step.

```

```cpp
// max_reduction.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Number of elements in the input array
#define N  (1 << 20)   // 1M elements
// Thread block size
#define BLOCK_SIZE 256

/* Kernel to perform reduction to find maximum within each block.
   Each block writes its maximum to the block_results array. */
__global__ void blockMaxReduce(const int *input, int *block_results, size_t n)
{
    // Shared memory for reduction
    __shared__ int sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory, or set to minimum if out of bounds
    int val = (idx < n) ? input[idx] : INT_MIN;
    sdata[tid] = val;
    __syncthreads();

    // Perform reduction in shared memory
    // Unrolling half of the block size steps
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0)
    {
        block_results[blockIdx.x] = sdata[0];
    }
}

/* Kernel to reduce block results to a single maximum value. */
__global__ void finalMaxReduce(int *block_results, int *final_result, size_t num_blocks)
{
    __shared__ int sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int val = (idx < num_blocks) ? block_results[idx] : INT_MIN;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // Write the global maximum to final_result
        atomicMax(final_result, sdata[0]);
    }
}

int main(void)
{
    int *h_data = (int *)malloc(N * sizeof(int));
    if (!h_data)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize data with random integers
    for (int i = 0; i < N; ++i)
        h_data[i] = rand() % 1000000;  // random numbers [0, 999999]

    int *d_input = nullptr;
    int *d_block_results = nullptr;
    int *d_final_result = nullptr;

    cudaMalloc((void **)&d_input, N * sizeof(int));
    cudaMalloc((void **)&d_block_results, ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(int));
    cudaMalloc((void **)&d_final_result, sizeof(int));

    cudaMemcpy(d_input, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_final_result, INT_MIN, sizeof(int));

    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    // First kernel: block-wise reduction
    blockMaxReduce<<<gridDim, blockDim>>>(d_input, d_block_results, N);
    cudaDeviceSynchronize();

    // Second kernel: reduce block results
    finalMaxReduce<<<gridDim, blockDim>>>(d_block_results, d_final_result, gridDim.x);
    cudaDeviceSynchronize();

    int host_final_max;
    cudaMemcpy(&host_final_max, d_final_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Compute max on host for verification
    int host_max = INT_MIN;
    for (int i = 0; i < N; ++i)
        if (h_data[i] > host_max)
            host_max = h_data[i];

    printf("GPU max  : %d\n", host_final_max);
    printf("CPU max  : %d\n", host_max);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_block_results);
    cudaFree(d_final_result);
    free(h_data);

    return 0;
}
```