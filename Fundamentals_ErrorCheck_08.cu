```cpp
/*
Aim of the program:
Run a program that has an out-of-bounds memory access inside the kernel (an asynchronous error) with and without proper post-launch error checking (`cudaDeviceSynchronize`). Observe that without it, the program may complete without any visible error, but potentially with incorrect results.

Thinking:
- Allocate a small device array (N = 10).
- Launch a kernel that writes to indices up to N+5, thereby accessing memory beyond the allocated region.
- Perform the launch twice:
    1. Without calling `cudaDeviceSynchronize()` after the kernel. Immediately check for errors with `cudaGetLastError()` and perform a `cudaMemcpy()` to bring data back to the host. The asynchronous out-of-bounds access will not be reported yet, so we expect no error and possibly corrupted data.
    2. With a `cudaDeviceSynchronize()` after the kernel launch. The error from the out-of-bounds write will be reported at this point.
- Print host data to see that values may be corrupted in the first case.
- The program demonstrates the importance of synchronizing after kernel launches to catch asynchronous errors such as out-of-bounds memory accesses.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that deliberately writes beyond the bounds of the allocated array.
__global__ void outOfBoundsKernel(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally write up to N+5 elements.
    if (idx < N + 5)
    {
        d_arr[idx] = idx;  // Out-of-bounds write when idx >= N
    }
}

int main()
{
    const int N = 10;               // Size of the device array
    const int OUT_BOUNDS = 5;       // Extra elements to write beyond bounds
    const int TOTAL = N + OUT_BOUNDS; // Total elements we will attempt to write

    int *d_arr = nullptr;
    cudaError_t err = cudaMalloc((void **)&d_arr, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Allocate host memory large enough to hold the potentially corrupted data.
    int h_arr[TOTAL];
    // Initialize host memory for clarity.
    for (int i = 0; i < TOTAL; ++i)
        h_arr[i] = -1;

    // ---------- Case 1: Launch kernel without synchronization ----------
    printf("\nCase 1: Launch kernel without cudaDeviceSynchronize()\n");
    outOfBoundsKernel<<<1, TOTAL>>>(d_arr, N);

    // Immediately check for errors (will not catch asynchronous errors).
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cudaGetLastError after kernel launch: %s\n", cudaGetErrorString(err));
    else
        printf("cudaGetLastError after kernel launch: CUDA_SUCCESS\n");

    // Copy back the first N elements (the allocated region).
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        printf("cudaMemcpy after kernel launch (no sync): %s\n", cudaGetErrorString(err));
    else
        printf("cudaMemcpy after kernel launch (no sync): CUDA_SUCCESS\n");

    // Print the host data to observe any corruption.
    printf("Host array after no-sync launch (first %d elements):\n", N);
    for (int i = 0; i < N; ++i)
        printf("h_arr[%d] = %d\n", i, h_arr[i]);

    // ---------- Case 2: Launch kernel with synchronization ----------
    // Reinitialize device array to zero for clarity.
    err = cudaMemset(d_arr, 0, N * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Reinitialize host memory.
    for (int i = 0; i < TOTAL; ++i)
        h_arr[i] = -1;

    printf("\nCase 2: Launch kernel with cudaDeviceSynchronize()\n");
    outOfBoundsKernel<<<1, TOTAL>>>(d_arr, N);

    // Synchronize to catch asynchronous errors.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("cudaDeviceSynchronize after kernel launch: %s\n", cudaGetErrorString(err));
    else
        printf("cudaDeviceSynchronize after kernel launch: CUDA_SUCCESS\n");

    // After synchronization, check last error again.
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("cudaGetLastError after cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
    else
        printf("cudaGetLastError after cudaDeviceSynchronize: CUDA_SUCCESS\n");

    // Copy back the first N elements.
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        printf("cudaMemcpy after kernel launch (with sync): %s\n", cudaGetErrorString(err));
    else
        printf("cudaMemcpy after kernel launch (with sync): CUDA_SUCCESS\n");

    // Print the host data to observe any corruption.
    printf("Host array after sync launch (first %d elements):\n", N);
    for (int i = 0; i < N; ++i)
        printf("h_arr[%d] = %d\n", i, h_arr[i]);

    // Clean up.
    cudaFree(d_arr);
    return 0;
}
```