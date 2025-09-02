```cpp
/*
Aim: Explain the difference between synchronous and asynchronous CUDA errors. 
Write a code snippet that would cause a synchronous error (e.g., invalid cudaMalloc size) 
and another that would cause an asynchronous error (e.g., out-of-bounds access inside a kernel).

Thinking:
1. CUDA operations can fail in two distinct ways:
   - Synchronous errors are reported immediately by the CUDA API function that was called.
     For example, cudaMalloc will return a non-zero error code if the requested size is
     invalid (negative, too large, or exceeds the device memory). These errors are
     detected on the host side before any kernel launch or asynchronous operation
     can take place.

   - Asynchronous errors happen inside a kernel or other asynchronous stream
     operations. The API call that initiates the operation (e.g., kernel launch)
     returns successfully, but the actual error (like out‑of‑bounds memory
     access, division by zero, etc.) occurs later when the device processes the
     instruction. The error is captured by cudaGetLastError() only after the
     operation has completed or when a synchronization point (cudaDeviceSynchronize
     or cudaStreamSynchronize) is reached.

2. In the code below:
   - The first part intentionally requests an unrealistically large allocation
     to trigger a synchronous cudaMalloc error. We check the return value
     immediately and print the error.

   - The second part allocates a small array, launches a kernel that writes
     past the end of the array (indexing beyond the allocated size). The
     kernel launch returns successfully; the error is captured only when
     cudaGetLastError() or cudaDeviceSynchronize() is called. We demonstrate
     both error retrieval methods.

3. The program is self-contained and compiles with nvcc. It prints both
   error messages to illustrate the difference between the two types.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple error checking macro for synchronous errors
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that will write out-of-bounds for indices >= N
__global__ void outOfBoundsKernel(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally write to d_arr even if idx >= N
    d_arr[idx] = idx * 2;
}

int main(void) {
    /* ---------- Synchronous error: invalid cudaMalloc size ---------- */
    size_t hugeSize = 1ULL << 62; // ~4EB, far beyond device memory
    int *d_ptr = nullptr;
    cudaError_t syncErr = cudaMalloc((void**)&d_ptr, hugeSize);
    if (syncErr != cudaSuccess) {
        printf("Synchronous error (invalid cudaMalloc size): %s\n",
               cudaGetErrorString(syncErr));
    } else {
        // Should never get here
        printf("Unexpectedly succeeded in allocating huge memory.\n");
        cudaFree(d_ptr);
    }

    /* ---------- Asynchronous error: out-of-bounds kernel access ---------- */
    const int N = 10;
    int *d_arr = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_arr, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_arr, 0, N * sizeof(int)));

    // Launch more threads than elements to force out-of-bounds write
    const int threadsPerBlock = 4;
    const int blocks = 4; // 4*4 = 16 threads > N=10
    outOfBoundsKernel<<<blocks, threadsPerBlock>>>(d_arr, N);

    // Check for errors immediately after launch (will still be asynchronous)
    cudaError_t asyncErr = cudaGetLastError();
    if (asyncErr != cudaSuccess) {
        printf("Asynchronous error after kernel launch (early check): %s\n",
               cudaGetErrorString(asyncErr));
    } else {
        printf("No error reported immediately after kernel launch.\n");
    }

    // Force synchronization to allow the device to process the kernel
    asyncErr = cudaDeviceSynchronize();
    if (asyncErr != cudaSuccess) {
        printf("Asynchronous error detected after cudaDeviceSynchronize: %s\n",
               cudaGetErrorString(asyncErr));
    } else {
        printf("Kernel executed successfully (unexpectedly).\n");
    }

    // Cleanup
    cudaFree(d_arr);

    return 0;
}
```