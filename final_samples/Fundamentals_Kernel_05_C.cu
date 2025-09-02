/*
Aim: Calculate the total number of threads on the host (gridDim.x * blockDim.x) and pass it to the kernel. Inside the kernel, print the global ID and the total number of threads.

Thinking:
- We need to launch a CUDA kernel from the host. The total number of threads is simply gridDim.x * blockDim.x for a 1‑D launch.
- This total should be computed on the host and then passed as a kernel argument.
- Inside the kernel, each thread calculates its global index: threadIdx.x + blockIdx.x * blockDim.x.
- Each thread prints its global ID along with the total thread count. Since `printf` is supported in device code, we can use it directly.
- We'll add basic error checking for CUDA API calls to make the program robust.
- The program will be self‑contained in a single .cu file and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Utility macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that prints the global thread ID and total threads
__global__ void print_thread_info(int total_threads) {
    // Compute global ID
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    // Print information
    printf("Thread %d of %d\n", global_id, total_threads);
}

int main(void) {
    // Define grid and block dimensions
    const int threads_per_block = 256; // typical value
    const int blocks = 10;             // arbitrary number of blocks

    // Compute total number of threads on the host
    int total_threads = threads_per_block * blocks;

    // Launch kernel
    print_thread_info<<<blocks, threads_per_block>>>(total_threads);

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset device (optional but good practice)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
