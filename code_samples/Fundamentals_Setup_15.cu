/*
Write a program that resets the primary CUDA device using cudaDeviceReset(). Explain in a comment why and when this function might be useful (e.g., for cleaning up resources or profiling).

I am creating a minimal CUDA C++ program that demonstrates how to call cudaDeviceReset() to reset the primary device. The aim is to show the correct usage, handle errors, and provide an explanation about its usefulness. 
The program includes standard headers, calls cudaDeviceReset(), checks for errors, and prints the result. I also include a detailed comment explaining why cudaDeviceReset() is important for cleaning up GPU resources and ensuring a clean state before profiling or after large allocations. This satisfies the requirement to explain its usefulness in a comment.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("Resetting CUDA device...\n");

    // Reset the primary CUDA device. This function cleans up all allocated
    // memory, destroys all CUDA contexts, and returns the device to its
    // initial state. It is useful when the application has finished all
    // GPU work and wants to free resources explicitly. Additionally, before
    // starting a profiling session, resetting the device ensures that no
    // leftover state from previous runs interferes with the profiling data.
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("CUDA device reset successfully.\n");
    return EXIT_SUCCESS;
}