/*
Aim: Demonstrate catching an asynchronous error from inside the kernel (e.g. an illegal memory access) using `CHECK(cudaDeviceSynchronize())`.

Thinking:
- We'll define a macro CHECK that calls a CUDA API and prints an error message if the call fails.
- Allocate a small array on the device (size 10) and launch a kernel that intentionally writes out of bounds.
- The illegal memory access will be asynchronous; the kernel launch itself will succeed, but the error will surface when we synchronize.
- By calling CHECK(cudaDeviceSynchronize()), we catch the asynchronous error and print the corresponding CUDA error code and message.
- This demonstrates how to catch asynchronous errors in CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                           \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s at line %d: %s (%d)\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that intentionally performs an illegal memory access
__global__ void badKernel(int *devArr, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Attempt to write beyond the bounds of the array
    if (idx == 0) {
        // size is 10; accessing index 100 is out-of-bounds
        devArr[100] = 42;  // Illegal write
    }
}

int main(void) {
    const int N = 10;
    int *devArr = NULL;

    // Allocate device memory
    CHECK(cudaMalloc((void**)&devArr, N * sizeof(int)));
    // (Optional) Initialize memory to zero
    CHECK(cudaMemset(devArr, 0, N * sizeof(int)));

    // Launch kernel with a single thread to trigger the illegal access
    badKernel<<<1, 1>>>(devArr, N);

    // Synchronize and check for errors
    // This call will catch the illegal memory access that happened asynchronously in the kernel
    CHECK(cudaDeviceSynchronize());

    // If we reach this point, no error was detected (unlikely in this demo)
    printf("Kernel executed successfully (unexpected).\n");

    // Clean up
    CHECK(cudaFree(devArr));
    return 0;
}
