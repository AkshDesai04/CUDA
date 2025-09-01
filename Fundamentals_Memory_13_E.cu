/*
Aim: Check for the `cudaErrorMemoryAllocation` error specifically by comparing the return value to the enum.

Thinking:
- The goal is to demonstrate how to detect a specific CUDA error, namely cudaErrorMemoryAllocation, by comparing the return value of a CUDA runtime API call to the enum value `cudaErrorMemoryAllocation`.
- We'll use `cudaMalloc` to attempt to allocate a large chunk of device memory that is intentionally too large for most GPUs. This should trigger a memory allocation failure.
- The returned error code from `cudaMalloc` is checked against `cudaErrorMemoryAllocation`. If it matches, we print a message indicating that a memory allocation error was detected. If the allocation succeeds (unlikely for such a large request), we print a success message. For any other error code, we print a generic error message along with the error string obtained from `cudaGetErrorString`.
- The program is self‑contained, written in CUDA C, and compiles to a .cu file. No additional files or dependencies are required beyond the CUDA toolkit.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    size_t largeSize = (size_t)1 << 40; // 1 TB, intentionally huge to trigger failure
    void *d_ptr = NULL;

    cudaError_t err = cudaMalloc(&d_ptr, largeSize);

    if (err == cudaSuccess) {
        printf("cudaMalloc succeeded (unexpected for 1 TB request).\n");
        // Free the memory if it somehow succeeded
        cudaFree(d_ptr);
    } else if (err == cudaErrorMemoryAllocation) {
        printf("cudaErrorMemoryAllocation detected: Unable to allocate %zu bytes on the device.\n", largeSize);
    } else {
        printf("cudaMalloc failed with error: %s\n", cudaGetErrorString(err));
    }

    // Reset the device (clean up any lingering state)
    cudaDeviceReset();

    return 0;
}
