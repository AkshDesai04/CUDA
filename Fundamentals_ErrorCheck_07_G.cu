/*
Is a kernel launch configuration error (e.g. too many threads) synchronous or asynchronous? (Asynchronous, but the error is usually available immediately via `cudaGetLastError`).

Thinking:
- Kernel launches are asynchronous: the host thread does not wait for the kernel to finish.
- Errors due to launch configuration (e.g., too many threads per block) are detected by the runtime at launch time.
- However, the error is not thrown until the host calls a CUDA runtime API that synchronizes the device or queries for errors.
- The standard way to check for such errors immediately after a launch is cudaGetLastError().
- This program queries the device properties to determine the maximum threads per block.
- It then intentionally launches a kernel with a block size exceeding this maximum to provoke a cudaErrorInvalidConfiguration.
- After the launch, it calls cudaGetLastError() and prints the error message, demonstrating that the error is available immediately.
- Finally, it calls cudaDeviceSynchronize() to ensure all previous operations are complete, and checks the error again (which should be cudaSuccess).
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void dummyKernel()
{
    // Empty kernel
}

int main()
{
    cudaDeviceProp devProp;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaGetDeviceProperties(&devProp, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Device %d: %s\n", device, devProp.name);
    printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);

    // Intentionally use too many threads per block to trigger an error
    int blockSize = devProp.maxThreadsPerBlock + 1000; // Exceeds limit
    int gridSize = 1;

    printf("\nLaunching kernel with blockSize=%d (exceeds max threads per block)\n", blockSize);
    dummyKernel<<<gridSize, blockSize>>>();

    // Immediately check for launch configuration errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaGetLastError() after launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaGetLastError() after launch: CUDA_SUCCESS\n");
    }

    // Now synchronize and check again
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize() error: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaDeviceSynchronize() succeeded.\n");
    }

    // Check error again after synchronization
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaGetLastError() after synchronization: %s\n", cudaGetErrorString(err));
    } else {
        printf("cudaGetLastError() after synchronization: CUDA_SUCCESS\n");
    }

    return EXIT_SUCCESS;
}
