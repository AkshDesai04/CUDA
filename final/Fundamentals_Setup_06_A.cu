/*
From the maximum dimensions [X, Y, Z], verify that X * Y * Z is not necessarily equal to `maxThreadsPerBlock` (since X, Y, and Z are individual dimension limits).

Thinking:
The CUDA programming model imposes two separate constraints for a kernel launch:
1. `maxThreadsPerBlock` – the maximum number of threads that a single block may contain (e.g., 1024 on many GPUs).
2. `maxThreadsDim` – three separate limits on the number of threads that can be assigned to each dimension of a block (e.g., 1024, 1024, 64 for x, y, z).

Because `maxThreadsDim` is per‑dimension, a block could theoretically use `maxThreadsDim.x * maxThreadsDim.y * maxThreadsDim.z` threads, which might exceed `maxThreadsPerBlock`. CUDA guarantees that the *product* of the thread counts in all three dimensions must not exceed `maxThreadsPerBlock`, but the individual dimension limits are independent. 

To demonstrate this, the program:
- Queries the device properties for `maxThreadsPerBlock` and `maxThreadsDim` (x, y, z).
- Prints these values and the product of the three dimension limits.
- Attempts to launch a trivial kernel using the maximum per‑dimension limits. If the product exceeds `maxThreadsPerBlock`, the launch will fail with an error, confirming that the product can indeed be larger than the block thread limit.

The program also includes simple CUDA error checking to capture and report the kernel launch status.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

// A trivial kernel that does nothing
__global__ void dummyKernel() {}

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Use the first device
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("Device: %s\n", prop.name);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  maxThreadsDim.x: %d\n", prop.maxThreadsDim.x);
    printf("  maxThreadsDim.y: %d\n", prop.maxThreadsDim.y);
    printf("  maxThreadsDim.z: %d\n", prop.maxThreadsDim.z);

    long long dimProduct = (long long)prop.maxThreadsDim.x *
                           (long long)prop.maxThreadsDim.y *
                           (long long)prop.maxThreadsDim.z;

    printf("  Product of maxThreadsDim: %lld\n", dimProduct);
    if (dimProduct > prop.maxThreadsPerBlock) {
        printf("  Note: The product of maxThreadsDim exceeds maxThreadsPerBlock.\n");
    } else {
        printf("  Note: The product of maxThreadsDim is within maxThreadsPerBlock.\n");
    }

    // Attempt to launch a kernel with block dimensions equal to maxThreadsDim
    dim3 blockDim(prop.maxThreadsDim.x, prop.maxThreadsDim.y, prop.maxThreadsDim.z);
    dim3 gridDim(1, 1, 1);

    printf("\nAttempting to launch kernel with blockDim = {%d, %d, %d}\n",
           blockDim.x, blockDim.y, blockDim.z);

    dummyKernel<<<gridDim, blockDim>>>();
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
    } else {
        // Wait for kernel to finish
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("Kernel launched successfully.\n");
    }

    return EXIT_SUCCESS;
}
