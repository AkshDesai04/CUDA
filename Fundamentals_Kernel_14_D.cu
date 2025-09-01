```cuda
/*
Aim of the program: 
Use `cudaDeviceSynchronize()` after the kernel launch. This is a common way to catch asynchronous errors from the kernel execution itself, not just launch configuration errors.

Thinking:
To demonstrate the importance of calling `cudaDeviceSynchronize()` after a kernel launch, we intentionally create a kernel that performs an out-of-bounds memory access. The kernel itself will launch without error, but the out-of-bounds write will be detected only when the device is synchronized. This example will:
1. Allocate device memory for an array of size N.
2. Launch a kernel that writes to index i for each thread i. For the last thread we force an out-of-bounds write by accessing index N (one past the end).
3. Call `cudaDeviceSynchronize()` to flush any asynchronous errors.
4. Check the return value of the synchronization call; if an error is detected, we print a helpful message.
5. Clean up device memory.

This program shows how `cudaDeviceSynchronize()` is essential for catching errors that happen during kernel execution, which are not caught by `cudaGetLastError()` immediately after the launch.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Helper macro to check CUDA errors
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Kernel that deliberately writes out of bounds for the last thread
__global__ void out_of_bounds_kernel(int *d_arr)
{
    int idx = threadIdx.x;
    // Intentional out-of-bounds write for idx == N-1
    if (idx == N-1)
        d_arr[idx + 1] = 42; // idx+1 == N -> out of bounds
    else
        d_arr[idx] = idx;
}

int main(void)
{
    int *d_arr = NULL;
    size_t size = N * sizeof(int);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_arr, size));

    // Launch kernel with N threads (single block for simplicity)
    out_of_bounds_kernel<<<1, N>>>(d_arr);

    // Check for launch configuration errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to catch asynchronous errors (e.g., out-of-bounds)
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess)
    {
        fprintf(stderr, "Asynchronous kernel error detected during cudaDeviceSynchronize(): %s\n",
                cudaGetErrorString(sync_err));
    }
    else
    {
        printf("Kernel executed successfully.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
```