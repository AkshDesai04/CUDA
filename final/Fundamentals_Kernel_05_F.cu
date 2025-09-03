/*
Aim: Launch with more threads than you need (e.g., for an array of size 100, launch 128 threads). Add a check `if (id < 100)` before printing or writing to memory.

Thinking:
The goal of this program is to demonstrate how to safely launch a CUDA kernel with more threads than the amount of work that needs to be performed. In this case, we have an array of size 100, but we will launch 128 threads. Each thread will compute its global index (`id`) and then check whether this index is less than 100 before performing any work (in this example, writing the index to an array). This guard ensures that the extra threads do not access or write out of bounds memory, which would lead to undefined behavior or a crash.

Implementation details:
- Define `N` as the logical size of the work (100).
- Allocate a device array of length `N`.
- Launch a single block with 128 threads (`dim3 block(128); dim3 grid(1);`).
- In the kernel, compute the global thread id (`id = blockIdx.x * blockDim.x + threadIdx.x`).
- Use `if (id < N)` guard before writing to the array.
- After kernel execution, copy the array back to host and print the contents to verify correctness.
- Include basic error checking for CUDA API calls to simplify debugging.

The program is written in pure C/C++ with CUDA extensions and should compile with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 100          // Logical size of the work
#define THREADS 128    // Launch more threads than needed

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Kernel that writes the thread id into the array, guarded by a boundary check
__global__ void write_indices(int *d_array, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        d_array[id] = id;  // Store the thread id as data
    }
}

int main(void) {
    int *h_array = NULL;
    int *d_array = NULL;
    size_t size = N * sizeof(int);

    // Allocate host memory
    h_array = (int *)malloc(size);
    if (h_array == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_array, size));

    // Launch kernel with more threads than required
    dim3 blockDim(THREADS, 1, 1);
    dim3 gridDim(1, 1, 1);
    write_indices<<<gridDim, blockDim>>>(d_array, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost));

    // Print the results
    printf("Array contents after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_array));
    free(h_array);

    // Reset device (optional)
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
