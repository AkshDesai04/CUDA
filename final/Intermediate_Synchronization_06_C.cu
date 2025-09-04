/*
Aim of the program: Run this kernel with `cuda-memcheck --tool synccheck`. Analyze its output.

Thinking:
- synccheck tool detects improper synchronization in CUDA kernels, such as missing __syncthreads() calls or conditional __syncthreads() that not all threads execute.
- To generate a detectable case, I create a kernel where threads in a block conditionally skip a __syncthreads() based on a value that depends on the thread index.
- The kernel will perform a simple operation before the conditional sync, then the sync, then a second operation. Threads that skip the sync will proceed incorrectly, leading to a synchronization error.
- In the host code, we launch the kernel and use cudaDeviceSynchronize() to ensure the kernel completes before exiting.
- Running `cuda-memcheck --tool synccheck ./a.out` will produce a diagnostic indicating a synchronization failure at the __syncthreads() point.
- The program is written in CUDA C (.cu file) and contains minimal error checking for clarity.

*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sync_error_kernel(int *data)
{
    // Each thread reads its value
    int idx = threadIdx.x;
    int val = data[idx];

    // First operation: increment the value
    val += 1;

    // Conditional synchronization: only even threads call __syncthreads()
    if ((idx & 1) == 0) {
        __syncthreads();
    }

    // Second operation: write back
    data[idx] = val;
}

int main()
{
    const int N = 256;
    size_t size = N * sizeof(int);

    int *h_data = (int*)malloc(size);
    int *d_data;

    // Initialize host data
    for (int i = 0; i < N; ++i)
        h_data[i] = i;

    // Allocate device memory
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel with one block of 256 threads
    sync_error_kernel<<<1, N>>>(d_data);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back (not used, but to keep code realistic)
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_data);
    free(h_data);

    return 0;
}
