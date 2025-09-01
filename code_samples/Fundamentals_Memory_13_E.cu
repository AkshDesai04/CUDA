/*
Aim: Check for the `cudaErrorMemoryAllocation` error specifically by comparing the return value to the enum.

Thinking:
We need a simple CUDA C program that demonstrates error checking for a specific CUDA error type.
The CUDA runtime API function `cudaMalloc` returns a value of type `cudaError_t`.  When the call fails
because the device cannot satisfy the requested allocation size, the function returns the enum value
`cudaErrorMemoryAllocation`.  The task is to explicitly compare the returned error code to that
enum and handle it separately from other possible errors.

The program will:
1. Attempt to allocate an extremely large amount of device memory so that the allocation will
   most likely fail on any reasonably sized GPU.
2. Capture the return value from `cudaMalloc`.
3. If the return value is not `cudaSuccess`, compare it explicitly to `cudaErrorMemoryAllocation`.
4. Print a specific message if the error is due to memory allocation, otherwise print a generic
   error message using `cudaGetErrorString`.
5. Clean up by resetting the device (if any memory was allocated).

The code is self‑contained, uses only standard CUDA runtime API calls, and can be compiled with
`nvcc`.  The large allocation request is intentional; it should trigger the specific error
condition on most devices, allowing us to observe the error handling logic.

The implementation below follows these steps and demonstrates the required comparison to the
`cudaErrorMemoryAllocation` enum value.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaError_t err;
    size_t large_size = (size_t)1ULL << 33; // Attempt to allocate 8 GiB (may fail)
    void *dev_ptr = NULL;

    printf("Attempting to allocate %zu bytes on the device...\n", large_size);

    err = cudaMalloc(&dev_ptr, large_size);

    if (err != cudaSuccess) {
        if (err == cudaErrorMemoryAllocation) {
            fprintf(stderr, "Error: cudaMalloc failed due to insufficient device memory.\n");
        } else {
            fprintf(stderr, "Error: cudaMalloc failed with error: %s\n", cudaGetErrorString(err));
        }
        // No need to free dev_ptr because allocation failed
    } else {
        printf("cudaMalloc succeeded. Memory allocated at device address %p.\n", dev_ptr);
        // Use the memory for some work here (omitted)

        // Free the allocated memory
        err = cudaFree(dev_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "Warning: cudaFree failed with error: %s\n", cudaGetErrorString(err));
        } else {
            printf("cudaFree succeeded.\n");
        }
    }

    // Reset the device to clean up any remaining state
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Warning: cudaDeviceReset failed with error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
