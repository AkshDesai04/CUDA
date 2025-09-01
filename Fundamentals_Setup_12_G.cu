/*
Pass a host pointer (e.g., from `malloc`) directly to a kernel that expects a device pointer.

The purpose of this program is to illustrate the incorrect usage of passing a host memory pointer directly to a CUDA kernel that expects a device memory pointer. In CUDA, memory allocated on the host (using `malloc` or `new`) resides in the CPU address space, while device memory allocated with `cudaMalloc` resides in the GPU address space. Attempting to dereference a host pointer from within a device kernel results in an invalid device pointer error and undefined behavior.

This example does the following:
1. Allocates an array of integers on the host using `malloc`.
2. Initializes the array with some values.
3. Launches a simple kernel that attempts to increment each element of the array. The kernel receives the host pointer as its argument, treating it as a device pointer.
4. After the kernel launch, we check for errors using `cudaGetLastError()` and `cudaDeviceSynchronize()`. This will catch the invalid pointer error.
5. Finally, we print the array contents to show that the kernel did not modify the data because the kernel failed.

To fix this issue, one would normally allocate memory on the device with `cudaMalloc`, copy the data from the host to the device with `cudaMemcpy`, launch the kernel with the device pointer, and then copy the results back to the host.

This program demonstrates why such an incorrect approach fails and serves as a cautionary example for CUDA developers.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Simple kernel that increments each element of an integer array */
__global__ void increment(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_arr[idx] += 1;   // Attempt to dereference host pointer as device pointer
    }
}

int main(void)
{
    const int N = 10;
    const size_t size = N * sizeof(int);

    /* Allocate array on host */
    int *h_arr = (int *)malloc(size);
    if (!h_arr)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (int i = 0; i < N; ++i)
    {
        h_arr[i] = i * 10;  // Just some sample values
    }

    printf("Host array before kernel launch:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    /* Launch kernel with host pointer (incorrect usage) */
    int threadsPerBlock = 4;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment<<<blocksPerGrid, threadsPerBlock>>>(h_arr, N);

    /* Check for errors immediately after launch */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    /* Synchronize to catch any asynchronous errors */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization error: %s\n", cudaGetErrorString(err));
    }

    /* Print host array after kernel launch to see if it was modified */
    printf("Host array after kernel launch:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    /* Clean up */
    free(h_arr);

    return 0;
}
