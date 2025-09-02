```cuda
/* 
Aim of the program: 
Experiment with kernel launch failure. Try to launch a kernel with more threads per block than the device maximum (e.g., 2048). Use proper error checking after the launch to catch and report the cudaErrorInvalidConfiguration error.

Thinking:
1. A simple CUDA program will query the device's maximum threads per block using cudaGetDeviceProperties.
2. We'll deliberately set the block size to 2048, which is typically above the maximum (commonly 1024).
3. The kernel will be a dummy kernel that does nothing; its purpose is only to trigger the launch configuration error.
4. After the launch, we use cudaGetLastError() to capture the error. If the error code is cudaErrorInvalidConfiguration, we print a clear message indicating that the launch failed due to an invalid configuration.
5. We also perform cudaDeviceSynchronize() to force any kernel launch errors that occur during execution to be reported.
6. Standard error checking macros are used to keep the code clean and readable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Dummy kernel that does nothing
__global__ void dummyKernel(void) {
    // No operation
}

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device %d: \"%s\"\n", device, prop.name);
    printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);

    // Intentionally exceed the maximum
    int threadsPerBlock = 2048;
    int numBlocks = 1;

    printf("Launching kernel with %d threads per block (exceeds maximum)...\n", threadsPerBlock);
    dummyKernel<<<numBlocks, threadsPerBlock>>>();

    // Check for launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        if (launchErr == cudaErrorInvalidConfiguration) {
            fprintf(stderr, "Launch failed: cudaErrorInvalidConfiguration (invalid configuration)\n");
        } else {
            fprintf(stderr, "Launch failed: %s\n", cudaGetErrorString(launchErr));
        }
    } else {
        // Even if the launch succeeded, synchronize to catch any errors that occur during execution
        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) {
            if (syncErr == cudaErrorInvalidConfiguration) {
                fprintf(stderr, "Execution failed: cudaErrorInvalidConfiguration (invalid configuration)\n");
            } else {
                fprintf(stderr, "Execution failed: %s\n", cudaGetErrorString(syncErr));
            }
        } else {
            printf("Kernel launched and executed successfully (unexpected).\n");
        }
    }

    return 0;
}
```