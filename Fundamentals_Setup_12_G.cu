/*
Aim: Pass a host pointer (e.g., from `malloc`) directly to a kernel that expects a device pointer.

Thinking: In CUDA, device memory and host memory are separate address spaces. Passing a host pointer directly
to a kernel that expects a device pointer will cause undefined behavior because the kernel will interpret
the host address as a device address. This example demonstrates the error by allocating an array on the
host with malloc, defining a kernel that operates on device memory, and then launching that kernel with
the host pointer. The program compiles, but at runtime it will likely crash or produce incorrect results,
highlighting the importance of using cudaMalloc for device allocations and cudaMemcpy to transfer data
between host and device. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add_one(int *d_array, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_array[idx] += 1;
    }
}

int main(void)
{
    const int N = 10;
    const size_t size = N * sizeof(int);

    /* Allocate host memory using malloc */
    int *h_array = (int *)malloc(size);
    if (h_array == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array */
    for (int i = 0; i < N; ++i) {
        h_array[i] = i;
    }

    /* Intentionally pass the host pointer to a kernel that expects a device pointer */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel with host pointer (undefined behavior) */
    add_one<<<blocksPerGrid, threadsPerBlock>>>(h_array, N);

    /* Check for launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    /* Wait for GPU to finish before accessing on host */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA device synchronize error: %s\n", cudaGetErrorString(err));
    }

    /* Print the host array after kernel execution (results are unreliable) */
    printf("Host array after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    /* Clean up */
    free(h_array);

    return 0;
}
