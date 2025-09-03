```cuda
/*
Aim of the program:
Check for the `cudaErrorMemoryAllocation` error specifically by comparing the return value to the enum.

Thinking:
The goal is to demonstrate how to detect a specific CUDA runtime error—cudaErrorMemoryAllocation—by inspecting the return value of a CUDA API call. 
We will attempt to allocate an overly large amount of GPU memory using cudaMalloc. 
If the allocation fails, the function returns an error code. By comparing this code to cudaErrorMemoryAllocation we can determine if the failure was due to insufficient memory. 
We’ll print out a message indicating the specific error type. The program will also handle other errors gracefully, printing a generic error message for non‑allocation failures. 
The code uses standard CUDA headers and a simple main function; it compiles with nvcc and can be run on any system with a CUDA-capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Attempt to allocate a huge amount of memory to trigger an allocation failure.
    size_t largeSize = (size_t)1 << 35; // 32 GiB (adjust as needed)
    void* devPtr = NULL;
    cudaError_t err = cudaMalloc(&devPtr, largeSize);

    if (err == cudaSuccess) {
        printf("Memory allocation succeeded unexpectedly.\n");
        // Free the allocated memory before exiting
        cudaFree(devPtr);
    } else if (err == cudaErrorMemoryAllocation) {
        printf("Error: cudaErrorMemoryAllocation detected. Requested size: %zu bytes.\n", largeSize);
    } else {
        printf("Error: cudaMalloc failed with error: %s\n", cudaGetErrorString(err));
    }

    // Always call cudaDeviceReset to clean up the device state
    cudaDeviceReset();

    return 0;
}
```