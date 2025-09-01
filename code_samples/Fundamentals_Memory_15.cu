```cuda
/*
Declare a global device variable using `__device__`. Write a kernel that modifies this variable. Write another kernel that reads it. Demonstrate how its value persists across kernel launches within the same application.

My approach:
1. Declare a global `__device__` integer variable `g_value` initialized to 0.
2. Write a kernel `modifyKernel` that simply assigns the passed value to `g_value`.
3. Write a kernel `readKernel` that reads `g_value` and writes it into a device array that the host can later copy back.
4. In `main`:
   - Allocate device memory for a single integer to receive the value.
   - Launch `modifyKernel` with a value (e.g., 42).
   - Launch `readKernel` to copy the current `g_value` into the device array.
   - Copy that value back to host and print it.
   - Repeat the process with a different value (e.g., 99) to show that the variable retains its value across kernel launches.
5. Include simple CUDA error checking to ensure correct execution.

This demonstrates that the global device variable `g_value` persists across multiple kernel invocations in the same program run.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple CUDA error-checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Global device variable
__device__ int g_value = 0;

// Kernel that modifies the global device variable
__global__ void modifyKernel(int val) {
    g_value = val;
}

// Kernel that reads the global device variable and writes it to an output array
__global__ void readKernel(int *out) {
    // Only one thread writes to the output
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = g_value;
    }
}

int main(void) {
    const int BLOCKS = 1;
    const int THREADS_PER_BLOCK = 1;

    int hostValue;
    int *devValue = nullptr;

    // Allocate device memory to receive the value read from g_value
    CUDA_CHECK(cudaMalloc((void**)&devValue, sizeof(int)));

    // First modification: set g_value to 42
    printf("Launching modifyKernel with value 42\n");
    modifyKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(42);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back the value
    printf("Launching readKernel to read g_value\n");
    readKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(devValue);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host and print
    CUDA_CHECK(cudaMemcpy(&hostValue, devValue, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Value read from g_value after first launch: %d\n", hostValue);

    // Second modification: set g_value to 99
    printf("\nLaunching modifyKernel with value 99\n");
    modifyKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(99);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back the value again
    printf("Launching readKernel to read g_value\n");
    readKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(devValue);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&hostValue, devValue, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Value read from g_value after second launch: %d\n", hostValue);

    // Clean up
    CUDA_CHECK(cudaFree(devValue));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```