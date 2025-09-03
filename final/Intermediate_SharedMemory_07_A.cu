```cuda
/*
Aim of the program: Generalize this to reverse a large array by having each block reverse its own segment.

Thinking:
1. We want to reverse a large array on the GPU, but we want to demonstrate how to partition the work across blocks.
2. Each block will be responsible for a contiguous segment of the array.  The segment length is defined by the
   grid configuration (block size and number of blocks).  Each thread in a block will compute its global index
   and then compute the partner index from the opposite end of the array.
3. By swapping an element with its symmetric partner only once (when the global index is less than the partner),
   we guarantee that the entire array is reversed.  The work is distributed across blocks, so each block
   operates on its own segment of indices.
4. The kernel uses simple arithmetic: `idx = blockIdx.x * blockDim.x + threadIdx.x;` gives the global
   index for the thread.  The partner index is `j = N - 1 - idx;`.  If `idx < j` we perform the swap.
   This avoids double swapping.
5. On the host side we allocate a host array, initialize it, copy it to device memory, launch the kernel,
   copy the result back, and print a small portion to verify the reversal.  Error checking is included
   after each CUDA API call.
6. The code is self‑contained in a single .cu file and can be compiled with `nvcc -o reverse reverse.cu`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel that reverses the array by swapping elements across the whole array.
// Each block works on its own segment of global indices.
__global__ void reverseArrayKernel(int *d_arr, int N) {
    // Global index of the thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Partner index from the opposite end
    int j = N - 1 - idx;
    // Swap only if idx < j to avoid double swaps
    if (idx < j) {
        int temp = d_arr[idx];
        d_arr[idx] = d_arr[j];
        d_arr[j] = temp;
    }
}

int main(void) {
    const int N = 1 << 20;          // Example: 1M integers
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_arr = (int *)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int BLOCK_SIZE = 256;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    reverseArrayKernel<<<gridSize, BLOCK_SIZE>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("First 10 elements after reversal:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    printf("Last 10 elements after reversal:\n");
    for (int i = N - 10; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));

    return 0;
}
```