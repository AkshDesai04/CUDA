```cpp
/*
After copying, use `cudaFree` to release the device memory.
Thinking: 
We want to demonstrate allocating device memory, copying data to it, launching a kernel, copying back, and freeing device memory. 
We'll allocate an array on host, copy to device using cudaMemcpy, launch a simple kernel that increments each element, copy back to host, print results, and finally free device memory with cudaFree, as requested. We'll also free host memory with free(). 
We will include basic error checking for CUDA calls. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel that adds 1 to each element of the array
__global__ void incrementKernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_arr[idx] += 1;
    }
}

int main(void)
{
    const int N = 10;
    const size_t size = N * sizeof(int);
    int *h_arr = (int *)malloc(size);
    if (h_arr == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i)
    {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void **)&d_arr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy from host to device
    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Print the results
    printf("Result after incrementing on device:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    err = cudaFree(d_arr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Free host memory
    free(h_arr);

    return EXIT_SUCCESS;
}
```