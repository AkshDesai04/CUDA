/*
Aim: Write to a negative index, like `d_ptr[-1]`, and check the report.

Thinking:
In CUDA, accessing memory outside the bounds of a device array leads to undefined behavior. The runtime does not
detect out‑of‑bounds writes or reads during normal execution; the only reliable way to catch such errors is to run the
program under `cuda-memcheck`. This example demonstrates how a kernel that writes to a negative index can be
compiled and executed, and how we can still inspect error codes using `cudaGetLastError()` and `cudaDeviceSynchronize()`
to see if any launch or synchronization failures occur. When executed with `cuda-memcheck`, the tool will report an
out‑of‑bounds write. The host code also copies the data back to verify that the expected values are unchanged
(aside from the intentional out‑of‑bounds write which will not be reflected in the allocated array).
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel that writes to a negative index (-1) and a valid index (0)
__global__ void negative_index_write(int *d_ptr, int size)
{
    // Only thread 0 performs the write for demonstration purposes
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Intentional out-of-bounds write
        d_ptr[-1] = 42;          // This writes before the start of the allocated array

        // Also write to a valid index for comparison
        if (size > 0)
            d_ptr[0] = 99;
    }
}

int main(void)
{
    const int N = 10;
    int *d_array = NULL;
    int *h_array = NULL;
    size_t bytes = N * sizeof(int);
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&d_array, bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Initialize device array to zeros
    err = cudaMemset(d_array, 0, bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Launch kernel with a single block and single thread
    negative_index_write<<<1, 1>>>(d_array, N);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel launched successfully (no launch error reported).\n");
    }

    // Synchronize and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel executed successfully (no runtime error reported).\n");
    }

    // Allocate host memory
    h_array = (int*)malloc(bytes);
    if (!h_array)
    {
        fprintf(stderr, "Host memory allocation failed.\n");
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Copy device data back to host
    err = cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_array);
        cudaFree(d_array);
        return EXIT_FAILURE;
    }

    // Print host array contents
    printf("Host array contents after kernel:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("h_array[%d] = %d\n", i, h_array[i]);
    }

    // Clean up
    free(h_array);
    cudaFree(d_array);

    return EXIT_SUCCESS;
}
