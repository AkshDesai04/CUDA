```cpp
/*
Aim of the program: How are synchronous errors reported? (Directly via the return code).

Thinking Process:
- The user wants a demonstration of how synchronous errors in CUDA are reported through the return code of CUDA API calls.
- A CUDA kernel launch can produce synchronous errors (e.g., launching with an invalid grid configuration). These errors are reported immediately by the launch API returning a non‑cudaSuccess code.
- Additionally, many CUDA runtime API functions (e.g., cudaMalloc, cudaMemcpy) return an error code directly. 
- We will write a small CUDA program that:
  1. Allocates device memory (synchronously).
  2. Launches a kernel with a deliberately incorrect configuration to trigger a synchronous error.
  3. Checks the return code of the kernel launch directly and reports it.
  4. Calls cudaGetLastError() after the launch to show the error message.
- We include a helper macro to check error codes and print meaningful messages.
- The program will compile as a .cu file and can be run with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA errors and print messages
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Simple kernel that does nothing
__global__ void dummyKernel(int *d_arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_arr[idx] = idx; // just a trivial operation
}

int main(void) {
    int *d_arr = NULL;
    const int N = 10;
    size_t size = N * sizeof(int);

    // Allocate device memory (synchronous error check)
    CHECK_CUDA(cudaMalloc((void**)&d_arr, size));

    // Intentionally launch kernel with zero threads per block to trigger synchronous error
    dim3 grid(1);
    dim3 block(0); // 0 threads is illegal

    // Launch the kernel and capture the return code
    cudaError_t launchErr = dummyKernel<<<grid, block>>>(d_arr);
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "Synchronous kernel launch error: %s\n",
                cudaGetErrorString(launchErr));
    } else {
        // If launch succeeded, synchronize and check for any asynchronous errors
        cudaError_t syncErr = cudaDeviceSynchronize();
        if (syncErr != cudaSuccess) {
            fprintf(stderr, "Asynchronous kernel execution error: %s\n",
                    cudaGetErrorString(syncErr));
        }
    }

    // Optionally, check cudaGetLastError() which returns the last error that occurred
    cudaError_t lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError reports: %s\n", cudaGetErrorString(lastErr));
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_arr));

    printf("Program completed.\n");
    return 0;
}
```