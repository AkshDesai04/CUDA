/*
Use `cudaDeviceSynchronize()` after the kernel launch. This is a common way to catch asynchronous errors from the kernel execution itself, not just launch configuration errors.

Thinking process:
1. We want to illustrate how an asynchronous error that occurs inside the kernel is only detected after a synchronization point, because kernel launches are asynchronous.
2. The simplest way to cause such an error is to intentionally write out of bounds in device memory. The launch configuration itself (grid/block dimensions, shared memory size, etc.) must be valid, otherwise `cudaGetLastError()` would catch it immediately.
3. The plan:
   - Allocate a device array with a small size (e.g., 5 integers).
   - Launch a kernel with 10 threads that writes to each index up to `size` (10), causing indices 5-9 to write beyond the allocated memory.
   - After the kernel launch, call `cudaGetLastError()` to confirm there is no launch error.
   - Then call `cudaDeviceSynchronize()`; this will force the host to wait for the kernel to finish and will surface any errors that occurred during execution (e.g., out‑of‑bounds access).
   - Capture the error code returned by `cudaDeviceSynchronize()` and print a human‑readable error string using `cudaGetErrorString`.
4. For clarity, the code also checks the result of the kernel launch and the subsequent synchronization, printing appropriate messages.
5. The code is self‑contained, compiles as a single .cu file, and demonstrates the usage of `cudaDeviceSynchronize()` for catching asynchronous kernel errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that writes to an array up to a given size.
__global__ void outOfBoundsKernel(int *arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = idx;  // Potential out‑of‑bounds write if arr is smaller than size.
    }
}

// Helper macro to check CUDA errors.
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void) {
    const int hostSize = 10;          // Number of elements we want the kernel to write.
    const int devSize  = 5;           // Actual size allocated on device (smaller to trigger OOB).

    int *d_arr = NULL;
    // Allocate device memory for devSize integers.
    CHECK_CUDA(cudaMalloc((void **)&d_arr, devSize * sizeof(int)));

    // Launch kernel with 10 threads (intentionally exceeding devSize).
    outOfBoundsKernel<<<1, 10>>>(d_arr, hostSize);

    // Check for launch configuration errors.
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(launchErr));
        // Clean up
        cudaFree(d_arr);
        return EXIT_FAILURE;
    } else {
        printf("Kernel launched successfully (no launch errors).\n");
    }

    // Synchronize to catch asynchronous errors (e.g., out‑of‑bounds writes).
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Asynchronous error detected after cudaDeviceSynchronize(): %s\n",
                cudaGetErrorString(syncErr));
    } else {
        printf("Kernel executed without asynchronous errors.\n");
    }

    // Clean up device memory.
    CHECK_CUDA(cudaFree(d_arr));

    return EXIT_SUCCESS;
}
