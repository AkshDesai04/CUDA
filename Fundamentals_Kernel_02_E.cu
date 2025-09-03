/*
Aim: Pass an integer as an argument to the kernel and print its value from the device.

Thinking:
To demonstrate how to pass an integer argument to a CUDA kernel and print it from the GPU, we will:
1. Define a simple kernel that takes an `int` parameter.
2. Inside the kernel, use `printf` (device printf) to output the integer value. Device printf requires the CUDA runtime to be compiled with `-arch=sm_20` or higher (most modern GPUs support this).
3. In the host code, we set an integer variable, launch the kernel with one block and one thread, and pass the integer by value.
4. After kernel launch, we synchronize the device to ensure the output is printed before the program exits.
5. For clarity and completeness, we include basic error checking for CUDA API calls using a helper macro.
6. The program is self‑contained in a single .cu file and can be compiled with `nvcc -o kernel_int kernel_int.cu`.

Note: This program uses only a single integer argument and prints it from a single thread for simplicity. For more complex use cases, you could pass pointers to device memory or use shared memory, but that is beyond the scope of this example.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that receives an integer and prints it
__global__ void printIntKernel(int value) {
    // Only one thread prints to avoid duplicate output
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Device received integer: %d\n", value);
    }
}

int main(void) {
    // Integer to pass to the kernel
    int hostValue = 42;

    // Launch configuration: 1 block, 1 thread
    dim3 blockDim(1);
    dim3 gridDim(1);

    // Launch kernel
    printIntKernel<<<gridDim, blockDim>>>(hostValue);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Wait for the device to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Optional: reset device
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
