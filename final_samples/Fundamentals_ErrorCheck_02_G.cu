/*
Does `cudaDeviceSynchronize` need to be wrapped? Yes, it can return asynchronous errors.

[Thinking: The program demonstrates that cudaDeviceSynchronize must be wrapped with error checking to capture asynchronous errors that occur during kernel execution. The kernel deliberately performs an out-of-bounds write, which results in an error that is only reported after synchronization. By checking the return value of cudaDeviceSynchronize, we can detect and report such errors. The code also includes a simple error-checking macro for clarity.]
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple error-checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel that deliberately writes out of bounds
__global__ void outOfBoundsKernel(int *arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Intentionally write to idx+1, causing the last thread to write out of bounds
        arr[idx + 1] = idx;
    }
}

int main() {
    const int N = 10;
    int *devData = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&devData, N * sizeof(int)));

    // Launch kernel with N threads (one per element)
    outOfBoundsKernel<<<1, N>>>(devData, N);

    // Wrap cudaDeviceSynchronize to catch the asynchronous error
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Asynchronous error detected after kernel launch: %s\n",
                cudaGetErrorString(syncErr));
    } else {
        printf("Kernel executed without detectable errors.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(devData));
    return 0;
}
