/*
At the end of the kernel, only thread 0 should write the final result from `shared_mem[0]` to the global memory output.

[Thinking]
- The goal is to perform a parallel reduction (sum) of an input array using shared memory.
- Each thread in the block loads multiple elements of the array in a strided manner and accumulates a local sum.
- These local sums are stored in shared memory.
- A standard tree-based reduction is then performed in shared memory.
- Finally, thread 0 writes the total sum from shared_mem[0] to the global output array.
- The kernel is launched with a single block so that only thread 0 writes to the output; this matches the requirement that only thread 0 performs the final write.
- The program includes a simple host main that allocates input data, copies it to the device, launches the kernel, and retrieves the result.
- Error checking macros are used for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256          // Number of threads per block
#define N 1024*1024             // Size of input array (1 million elements)

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that sums an array using shared memory
__global__ void reduceSum(const float *input, float *output, int n)
{
    __shared__ float shared_mem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Each thread computes a partial sum over strided elements
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        sum += input[i];
    }

    // Store partial sum in shared memory
    shared_mem[tid] = sum;
    __syncthreads();

    // Tree-based reduction within the block
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 writes the final result to global memory
    if (tid == 0) {
        output[0] = shared_mem[0];
    }
}

int main(void)
{
    // Allocate host memory
    float *h_input = (float *)malloc(N * sizeof(float));
    float h_output = 0.0f;

    // Initialize input with random numbers
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_input = NULL;
    float *d_output = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_output, sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with one block and BLOCK_SIZE threads
    reduceSum<<<1, BLOCK_SIZE>>>(d_input, d_output, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result
    printf("Sum of array elements: %f\n", h_output);

    // Clean up
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);

    return 0;
}
