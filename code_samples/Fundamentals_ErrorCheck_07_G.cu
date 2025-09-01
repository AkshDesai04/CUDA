/*
Is a kernel launch configuration error (e.g. too many threads) synchronous or asynchronous? (Asynchronous, but the error is usually available immediately via cudaGetLastError).
This program demonstrates that a kernel launch with an invalid configuration (too many threads per block) triggers an asynchronous error, but the error can be queried immediately with cudaGetLastError(). The program:
- Queries the device to find the maximum threads per block.
- Intentionally launches a kernel with more threads per block than allowed.
- Immediately checks cudaGetLastError() to retrieve the error code.
- Prints the error string and the error code.
- Calls cudaDeviceSynchronize() to observe that the error does not affect synchronization.
- Cleans up and exits.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // This kernel intentionally does nothing.
}

int main()
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    printf("Device: %d (%s)\n", device, prop.name);
    printf("Maximum threads per block supported: %d\n", maxThreadsPerBlock);

    // Intentionally set block dimension larger than maximum allowed
    int blockDim = maxThreadsPerBlock + 256; // exceed limit
    int gridDim = 1;

    printf("Launching kernel with blockDim = %d (exceeds limit)\n", blockDim);

    dummyKernel<<<gridDim, blockDim>>>();

    // Immediately query for error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error after kernel launch: %s (code %d)\n", cudaGetErrorString(err), err);
    } else {
        printf("No error reported by cudaGetLastError() immediately after launch.\n");
    }

    // Attempt to synchronize to see if any error propagates
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error during cudaDeviceSynchronize(): %s (code %d)\n", cudaGetErrorString(err), err);
    } else {
        printf("cudaDeviceSynchronize() completed without error.\n");
    }

    return 0;
}
