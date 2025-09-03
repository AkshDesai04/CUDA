```cuda
/*
Aim of the program: 
Write a kernel `increment_counter()` that has each thread perform `atomicAdd(&my_global_counter, 1);`. 
Launch it, then launch a second kernel `read_counter(int* out)` that has thread 0 write the value of 
`my_global_counter` to an output variable for verification.

Thinking process:
1. The global counter must reside in device memory.  We declare it as a `__device__` variable so that all
   kernels can access it.  Because device symbols cannot be initialized directly from host code,
   we copy a zero value into it before launching the kernels.
2. The `increment_counter` kernel is very simple: each thread just performs an `atomicAdd` on the global
   counter.  The kernel launch configuration (blocks and threads per block) determines how many
   increments are performed.
3. The `read_counter` kernel must only write the final counter value from a single thread (thread 0)
   into a device pointer provided by the host.  This is then copied back to the host for verification.
4. In `main` we allocate device memory for the output variable, set the global counter to zero,
   launch the two kernels, copy the result back, and print it.
5. Basic error checking after each CUDA API call and kernel launch ensures that any failures are
   reported.

The final code below is a self‑contained .cu file that follows the above plan and can be compiled
with `nvcc`.
*/
#include <stdio.h>
#include <cuda_runtime.h>

// Device global counter
__device__ int my_global_counter = 0;

// Kernel to increment the global counter atomically
__global__ void increment_counter()
{
    atomicAdd(&my_global_counter, 1);
}

// Kernel to read the global counter value into host-provided output
__global__ void read_counter(int *out)
{
    if (threadIdx.x == 0) {
        *out = my_global_counter;
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main()
{
    const int blocks = 10;
    const int threadsPerBlock = 256;

    // Allocate device memory for output
    int *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(int)));

    // Initialize global counter to zero
    int zero = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(my_global_counter, &zero, sizeof(int)));

    // Launch increment kernel
    increment_counter<<<blocks, threadsPerBlock>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Launch read kernel
    read_counter<<<1, 1>>>(d_out);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    int h_out = 0;
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify the result
    int expected = blocks * threadsPerBlock;
    printf("Final counter value: %d (expected %d)\n", h_out, expected);
    if (h_out == expected) {
        printf("Success: Counter incremented correctly.\n");
    } else {
        printf("Error: Counter value does not match expected.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
```