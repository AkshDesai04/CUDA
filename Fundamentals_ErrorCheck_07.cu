```cpp
/* 
Aim of the program (as requested in the prompt, word for word):

Explain the difference between synchronous and asynchronous CUDA errors. Write a code snippet that would cause a synchronous error (e.g., invalid `cudaMalloc` size) and another that would cause an asynchronous error (e.g., out-of-bounds access inside a kernel).

Thought process and design decisions:
- A synchronous CUDA error occurs immediately when a CUDA Runtime API call fails. For example, `cudaMalloc` will return an error code before the function returns if the requested allocation size is too large or otherwise invalid. This type of error can be caught by checking the return value of the function call right away.
- An asynchronous CUDA error occurs after the kernel has started executing or after the device has completed some work. Kernel launches are asynchronous; they return immediately. Errors that happen inside the kernel (such as out‑of‑bounds memory accesses) are not reported until we synchronize the device (e.g., `cudaDeviceSynchronize`) or query the last error with `cudaGetLastError`. Therefore, the error is not caught at the point of launch but only later.
- To illustrate these differences, the program will:
  1. Attempt a large `cudaMalloc` that is guaranteed to fail. We capture and print the error immediately – this demonstrates a synchronous error.
  2. Allocate a small array, launch a kernel that writes past the end of the array, then check for errors in two ways:
     - Right after the launch, using `cudaGetLastError` – this will report `cudaSuccess` because the launch itself succeeded.
     - After `cudaDeviceSynchronize`, which will force the kernel to finish and report any runtime errors – this will show the out‑of‑bounds error, illustrating an asynchronous error.
- The code is self‑contained and can be compiled with `nvcc` as a single `.cu` file. It prints descriptive messages for each error condition.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple error checking macro for synchronous calls
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error [%s:%d]: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel that writes out of bounds
__global__ void out_of_bounds_kernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally access one element past the end
    data[idx] = idx;
}

int main() {
    // 1. Synchronous error: attempt an impossible allocation size
    size_t hugeSize = static_cast<size_t>(1) << 60; // 1 exabyte
    int *d_ptr = nullptr;
    cudaError_t syncErr = cudaMalloc((void**)&d_ptr, hugeSize);
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "[Synchronous Error] cudaMalloc failed with: %s\n",
                cudaGetErrorString(syncErr));
    } else {
        // If by some miracle it succeeded, free it
        cudaFree(d_ptr);
    }

    // 2. Asynchronous error: out-of-bounds kernel access
    const int N = 16;
    int *d_small = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_small, N * sizeof(int)));

    // Launch kernel with N threads (each thread writes to its own index)
    out_of_bounds_kernel<<<1, N>>>(d_small, N);

    // Right after launch: check launch status
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "[Async Error] Kernel launch failed: %s\n",
                cudaGetErrorString(launchErr));
    } else {
        fprintf(stdout, "[Async Check] Kernel launch succeeded (no error reported yet).\n");
    }

    // Now synchronize to catch any errors that happened during execution
    cudaError_t syncErr2 = cudaDeviceSynchronize();
    if (syncErr2 != cudaSuccess) {
        fprintf(stderr, "[Async Error] After synchronization: %s\n",
                cudaGetErrorString(syncErr2));
    } else {
        fprintf(stdout, "[Async Check] Kernel executed successfully (no out-of-bounds error).\n");
    }

    // Clean up
    cudaFree(d_small);
    return 0;
}
```