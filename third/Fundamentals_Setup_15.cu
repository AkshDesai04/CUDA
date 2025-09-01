/*
Aim: Write a program that resets the primary CUDA device using `cudaDeviceReset()`. Explain in a comment why and when this function might be useful (e.g., for cleaning up resources or profiling).

Thought Process:
1. The goal is a minimal CUDA C program that demonstrates calling cudaDeviceReset() to reset the default device.
2. We need to include <cuda_runtime.h> and use C style (int main()).
3. We'll call cudaDeviceReset() and check the returned cudaError_t for errors.
4. A brief explanation comment will be added to describe why and when cudaDeviceReset() is useful:
   - Cleaning up all resources (memory, contexts, streams) on the device before the program exits.
   - For profiling or debugging, forcing a fresh device state between runs.
5. A simple printf will inform the user of success or failure.
6. The code will compile as a .cu file and can be built with nvcc.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Reset the primary CUDA device to clean up all resources.
    // This function is useful when you want to ensure that all GPU memory,
    // contexts, and other allocations are freed before the program exits.
    // It is also handy during profiling or debugging sessions where
    // you want a fresh device state for each run.
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("cudaDeviceReset succeeded.\n");
    return EXIT_SUCCESS;
}
