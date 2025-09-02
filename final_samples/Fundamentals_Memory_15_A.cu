/*
Initialize the `__device__` variable at declaration, e.g., `__device__ int my_global_counter = 0;`.

Thinking:
1. The task is to demonstrate how a __device__ variable can be initialized at declaration time.
2. We choose a simple integer counter, `my_global_counter`, and initialize it to 0.
3. A kernel will increment this counter atomically for each thread to show that the initialization works.
4. After kernel execution, we copy the counter value back to host memory using `cudaMemcpyFromSymbol` and print it.
5. The code includes basic error checking for CUDA API calls to ensure robustness.
6. This program is self-contained, uses only standard CUDA runtime APIs, and can be compiled with `nvcc` to produce an executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Initialize the __device__ variable at declaration
__device__ int my_global_counter = 0;

// Kernel that increments the global counter atomically
__global__ void increment_counter(int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        atomicAdd(&my_global_counter, 1);
    }
}

// Simple error-checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void)
{
    const int num_threads = 1000;
    const int threads_per_block = 256;
    const int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    increment_counter<<<num_blocks, threads_per_block>>>(num_threads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the device counter back to host
    int host_counter = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&host_counter,
                                    my_global_counter,
                                    sizeof(int),
                                    0,
                                    cudaMemcpyDeviceToHost));

    printf("my_global_counter = %d (expected %d)\n", host_counter, num_threads);

    return 0;
}
