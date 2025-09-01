```cpp
/*
Launch with more threads than you need (e.g., for an array of size 100, launch 128 threads). Add a check `if (id < 100)` before printing or writing to memory.

Thought process:
- The goal is to demonstrate a common CUDA pattern: launching more threads than the data size and guarding against out-of-bounds accesses.
- We'll create an array of 100 integers on the device, launch 128 threads (more than needed).
- Inside the kernel, each thread will compute its global index `id`. If `id < 100`, it will write its `id` value into the array; otherwise it will do nothing.
- After kernel execution, we copy the array back to the host and print its contents to verify correctness.
- We'll use standard CUDA error checking macros for robustness.
- The code will be a complete, self-contained .cu file that can be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel that writes its thread id into array if within bounds
__global__ void write_ids_kernel(int *d_arr, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        d_arr[id] = id;
    }
}

int main(void)
{
    const int N = 100;          // Size of the array
    const int THREADS_PER_BLOCK = 32;
    const int NUM_BLOCKS = 4;   // 4 blocks * 32 threads = 128 threads > N

    // Allocate host array
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device array
    int *d_arr = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    // Launch kernel
    write_ids_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_arr, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back to host
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("First 10 elements of the array:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));

    return 0;
}
```