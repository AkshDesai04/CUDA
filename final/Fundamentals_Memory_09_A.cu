```cu
/*
 * Aim of the program (word for word):
 * Explain in a comment why pinned memory transfers are faster. (Hint: DMA, no intermediate staging buffer).
 *
 * My thinking:
 * Pinned (page-locked) host memory allows the GPU DMA engine to directly access the memory region without going through the OS page tables, which means the CPU doesn't need to allocate a temporary staging buffer. This eliminates the extra copy step and reduces CPU overhead. Because the GPU can stream data directly, the transfer bandwidth is higher and latency lower. Also, pinned memory can be overlapped with kernel execution more efficiently due to its stable location in physical memory. In this program, we allocate a large array in pinned host memory, copy it to device memory, run a simple kernel, and copy the result back. The comments inside explain why the pinned allocation is critical for fast transfers.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that increments each element
__global__ void incrementKernel(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_arr[idx] += 1;
    }
}

int main()
{
    const int N = 1 << 20;           // 1M integers
    const size_t size = N * sizeof(int);

    // Allocate pinned (page-locked) host memory
    int *h_arr;
    cudaError_t err = cudaMallocHost((void**)&h_arr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i)
        h_arr[i] = i;

    // Allocate device memory
    int *d_arr;
    err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(h_arr);
        return 1;
    }

    // Copy from pinned host to device
    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFreeHost(h_arr);
        return 1;
    }

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFreeHost(h_arr);
        return 1;
    }

    // Copy result back to pinned host memory
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFreeHost(h_arr);
        return 1;
    }

    // Verify a few results
    for (int i = 0; i < 10; ++i)
        printf("h_arr[%d] = %d\n", i, h_arr[i]);

    // Clean up
    cudaFree(d_arr);
    cudaFreeHost(h_arr);

    return 0;
}
```