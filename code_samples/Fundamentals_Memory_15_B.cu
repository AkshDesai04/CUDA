/*
Aim: Write a kernel `increment_counter()` that has each thread perform `atomicAdd(&my_global_counter, 1);`. Launch it, then launch a second kernel `read_counter(int* out)` that has thread 0 write the value of `my_global_counter` to an output variable for verification.

Thinking:
- The global counter should be a device variable so that all threads can access and modify it via `atomicAdd`. We'll declare it as `__device__ int my_global_counter = 0;`.
- `increment_counter` will simply call `atomicAdd(&my_global_counter, 1);` for each thread. No synchronization inside the kernel is needed because `atomicAdd` handles concurrent updates.
- After launching `increment_counter`, we need to read the counter value back to host. A second kernel `read_counter` is used, with a single thread (thread 0) writing the current value of `my_global_counter` to a device pointer `out`. This demonstrates the counter’s value on the device and isolates the read to a single thread to avoid race conditions.
- The host code will:
  1. Allocate device memory for the output integer.
  2. Launch `increment_counter` with a chosen number of blocks and threads (e.g., 10 blocks of 256 threads gives 2560 increments).
  3. Synchronize to ensure the first kernel completes.
  4. Launch `read_counter` with one block and one thread.
  5. Synchronize again.
  6. Copy the result back to host and print it.
- A simple CUDA error‑checking macro (`CHECK_CUDA`) will make debugging easier.
- Finally, clean up allocated resources before exiting.

The program is self‑contained, written in C/C++ with CUDA extensions, and can be compiled with `nvcc`. It demonstrates atomic addition across many threads and retrieval of the final counter value.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Global counter accessible by all threads
__device__ int my_global_counter = 0;

// Kernel that increments the global counter atomically
__global__ void increment_counter()
{
    atomicAdd(&my_global_counter, 1);
}

// Kernel that writes the current value of the global counter to `out`
__global__ void read_counter(int* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *out = my_global_counter;
    }
}

int main()
{
    // Define launch configuration
    const int blocks = 10;
    const int threadsPerBlock = 256;
    const int totalThreads = blocks * threadsPerBlock; // Expected final counter value

    // Allocate device memory for the output variable
    int *d_out = NULL;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(int)));

    // Launch the increment kernel
    increment_counter<<<blocks, threadsPerBlock>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Launch the read kernel
    read_counter<<<1, 1>>>(d_out);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    int h_out = 0;
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Print the result
    printf("Final counter value: %d (expected %d)\n", h_out, totalThreads);

    // Clean up
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
