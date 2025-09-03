/*
Aim: Explain in a comment why memory leaks are bad, especially on a shared resource like a GPU.

[Thinking:
Memory leaks occur when a program allocates memory but fails to release it after use. 
On a GPU, which often serves multiple concurrent processes or applications, memory is a scarce, shared resource. 
When a leak happens, the allocated memory remains occupied for the lifetime of the process or even longer if the process does not terminate cleanly. 
This reduces the amount of memory available for other kernels or workloads, potentially causing out-of-memory errors, degrading performance, or forcing the system to swap or evict data from faster memory to slower storage. 
Because GPUs are typically designed to handle large parallel workloads, a small memory leak can quickly grow into a large unused allocation, leading to inefficiencies, increased power consumption, and instability in multi-tenant or shared environments. 
Proper memory management—freeing device memory with cudaFree, checking for errors, and using RAII or similar patterns—is essential to maintain the health and performance of GPU-accelerated applications and the overall system.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(int *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        data[idx] = idx;
    }
}

int main()
{
    const int N = 1024;
    int *d_data;
    cudaError_t err = cudaMalloc(&d_data, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    kernel<<<(N + 255) / 256, 256>>>(d_data);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    int h_data[N];
    err = cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Print first ten values
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Free the allocated device memory
    err = cudaFree(d_data);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
