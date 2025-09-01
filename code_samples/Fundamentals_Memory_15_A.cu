/*
Aim: Initialize the `__device__` variable at declaration, e.g., `__device__ int my_global_counter = 0;`.

Thinking:
The goal is to demonstrate that a device global variable can be initialized at the point of declaration and that this initialization persists across kernel launches. 
We declare a `__device__` integer named `my_global_counter` and initialize it to zero directly in its declaration. 
A simple kernel will atomically add a small value to this counter, allowing us to see that the variable started at zero and that updates made in device code are reflected back to the host. 
After launching the kernel, we copy the device variable back to a host variable using `cudaMemcpyFromSymbol` and print its value. 
If the counter is correctly initialized, its value after the kernel will be the sum of the increments performed by all threads. 
This program uses atomic addition to avoid race conditions and includes basic error checking for CUDA API calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device global variable initialized at declaration
__device__ int my_global_counter = 0;

// Kernel that atomically increments the device global counter
__global__ void inc_kernel(int inc_val)
{
    // Each thread increments the counter by inc_val
    atomicAdd(&my_global_counter, inc_val);
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    const int threads_per_block = 256;
    const int num_blocks = 4;
    const int inc_per_thread = 1;  // Each thread will add 1

    // Launch the kernel
    inc_kernel<<<num_blocks, threads_per_block>>>(inc_per_thread);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the device global variable back to host
    int host_counter = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&host_counter,
                                    my_global_counter,
                                    sizeof(int),
                                    0,
                                    cudaMemcpyDeviceToHost));

    printf("Global counter value after kernel launch: %d\n",
           host_counter);

    // Expected value: threads_per_block * num_blocks * inc_per_thread
    int expected = threads_per_block * num_blocks * inc_per_thread;
    printf("Expected value: %d\n", expected);

    return 0;
}
