```cuda
/*
Aim: Initialize the __device__ variable at declaration, e.g., __device__ int my_global_counter = 0;

Thinking:
1. We need a complete CUDA C program (.cu) that demonstrates initializing a __device__ global variable
   at its declaration. The example will declare a __device__ int called `my_global_counter` and set it
   to 0.
2. To show that the variable is indeed initialized, we will launch a simple kernel that increments
   this counter using `atomicAdd`. Each thread will add 1, ensuring no race conditions.
3. After the kernel execution we will copy the value of the device variable back to the host
   using `cudaMemcpyFromSymbol`, and then print it. The expected output will be the number of
   threads launched, confirming the initial zero value and subsequent increments.
4. The program will include basic error checking using a macro to simplify CUDA API calls.
5. The code will be fully self‑contained, compilable with `nvcc`, and will not rely on any external
   headers or libraries beyond the standard CUDA runtime.
6. The focus remains on the initialization syntax `__device__ int my_global_counter = 0;` and
   demonstrating its effect during runtime.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Macro for error checking
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Initialize the __device__ variable at declaration
__device__ int my_global_counter = 0;

// Kernel that increments the global counter
__global__ void increment_counter(int num_increments) {
    // Each thread increments the counter once
    atomicAdd(&my_global_counter, 1);
}

int main() {
    const int threads_per_block = 256;
    const int num_blocks = 1;
    const int total_threads = threads_per_block * num_blocks;

    // Launch kernel
    increment_counter<<<num_blocks, threads_per_block>>>(total_threads);
    CHECK_CUDA(cudaGetLastError()); // Check launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for completion

    // Copy device variable back to host
    int host_counter = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&host_counter, my_global_counter,
                                    sizeof(int), 0, cudaMemcpyDeviceToHost));

    printf("Final value of my_global_counter: %d (expected %d)\n",
           host_counter, total_threads);

    // Reset device counter for cleanliness (optional)
    int zero = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(my_global_counter, &zero,
                                  sizeof(int), 0, cudaMemcpyHostToDevice));

    return 0;
}
```