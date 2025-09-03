/*
Demonstrate its use by launching a kernel with an invalid configuration (e.g., too many threads) and catching the error with `CHECK(cudaPeekAtLastError())`.

Thinking:
The goal is to show how to intentionally trigger a CUDA runtime error by providing an invalid launch configuration, specifically by specifying more threads per block than the device allows (commonly 1024). After launching the kernel, we immediately call `cudaPeekAtLastError()` to retrieve the most recent error. A custom macro `CHECK` is defined to wrap this call, printing the error message but not terminating the program so the demonstration can be observed. The kernel itself is a simple empty kernel; its execution is irrelevant because the launch configuration is already invalid. The program prints informative messages before and after the kernel launch to illustrate the sequence of events. The code is written in C/C++ with CUDA and saved as a `.cu` file, ready to compile with `nvcc`.*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // Empty kernel – does nothing.
}

// Simple error-checking macro that prints the error but does not exit.
#define CHECK(call)                                           \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
        }                                                     \
    } while (0)

int main(void)
{
    // Attempt to launch a kernel with an invalid configuration:
    // Too many threads per block (e.g., 2000 > maximum 1024).
    int threadsPerBlock = 2000;  // Invalid value
    int blocksPerGrid    = 1;

    printf("Launching kernel with %d threads per block (invalid configuration)\n",
           threadsPerBlock);

    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>();

    // Catch any launch error using cudaPeekAtLastError().
    CHECK(cudaPeekAtLastError());

    // Optionally synchronize to catch any deferred errors.
    CHECK(cudaDeviceSynchronize());

    printf("Program completed.\n");

    // Clean up the device before exiting.
    CHECK(cudaDeviceReset());

    return 0;
}
