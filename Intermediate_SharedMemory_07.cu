/*
Aim: Write a CUDA kernel that reverses an array within a single block. Each thread i loads input[i] into shared_mem[i]. After a __syncthreads(), thread i reads from shared_mem[BLOCK_SIZE-1-i] and writes to output[i]. This is much faster than global memory reversal.

Thinking:
The problem requires a kernel that performs an in‑block array reversal using shared memory. The key steps are:
1. Each thread reads its corresponding element from the global input array into a shared memory buffer.
2. All threads synchronize so that the shared memory is fully populated.
3. Each thread then reads from the mirrored index in shared memory and writes that value to its output location in global memory.

Because shared memory is fast local memory, this approach reduces global memory traffic and is particularly efficient when the entire array fits within a single block. For demonstration purposes, we will:
- Define a constant BLOCK_SIZE (e.g., 256).
- Allocate device memory for an input and output array of length N ≤ BLOCK_SIZE.
- Launch the kernel with one block of BLOCK_SIZE threads.
- After the kernel completes, copy the result back to the host and print it to verify correctness.

We will also include a simple CUDA error‑checking macro to aid debugging. The code is written in standard C/C++ with CUDA extensions and can be compiled with `nvcc`. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Must be ≥ the size of the array to reverse

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that reverses an array within a single block
__global__ void reverseArray(const int *input, int *output, int N) {
    // Allocate shared memory for the block
    __shared__ int shared_mem[BLOCK_SIZE];

    int idx = threadIdx.x;

    // Load data into shared memory if within bounds
    if (idx < N) {
        shared_mem[idx] = input[idx];
    }

    __syncthreads();

    // Write the reversed element to the output array
    if (idx < N) {
        output[idx] = shared_mem[N - 1 - idx];
    }
}

int main(void) {
    const int N = 100; // Size of the array to reverse (must be <= BLOCK_SIZE)
    size_t size_bytes = N * sizeof(int);

    // Allocate host arrays
    int *h_input = (int *)malloc(size_bytes);
    int *h_output = (int *)malloc(size_bytes);

    // Initialize input array with sequential values
    for (int i = 0; i < N; ++i) {
        h_input[i] = i;
    }

    // Device pointers
    int *d_input = NULL;
    int *d_output = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_input, size_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_output, size_bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size_bytes, cudaMemcpyHostToDevice));

    // Launch kernel with one block
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(1);
    reverseArray<<<gridDim, blockDim>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size_bytes, cudaMemcpyDeviceToHost));

    // Print input and output for verification
    printf("Input Array  : ");
    for (int i = 0; i < N; ++i) printf("%d ", h_input[i]);
    printf("\n");

    printf("Reversed Array: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_output[i]);
    printf("\n");

    // Cleanup
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
