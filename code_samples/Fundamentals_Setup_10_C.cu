```cuda
/*
Aim of the program:
Add a comment explaining the role of the L2 cache in a GPU's memory hierarchy.

Thinking:
The program is a minimal CUDA example that demonstrates the memory hierarchy
in a GPU, specifically highlighting the role of the L2 cache.  The L2 cache
serves as a shared, larger cache between all streaming multiprocessors (SMs).
It sits between the device’s global memory (DRAM) and the L1 caches (or
per‑SM local caches) that each SM uses.  When a thread block accesses data
in global memory, the L1 cache first checks if the data is present.  If it
is not found in L1, the request goes up to L2.  L2 aggregates requests
from all SMs, improving bandwidth utilization and reducing contention
by sharing data that is used by multiple SMs.  This hierarchical caching
structure reduces latency and improves throughput for memory‑bound
kernels.  In this example, we launch a simple kernel that writes to
global memory; the compiler and driver will exercise the L2 cache when
the kernel accesses the same memory locations across different blocks.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void increment_kernel(int *d_data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_data[idx] += 1;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_data = (int *)malloc(size);
    for (int i = 0; i < N; ++i)
        h_data[i] = i;

    // Allocate device memory
    int *d_data;
    cudaMalloc((void **)&d_data, size);

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Copy result back to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Verify result (simple check)
    if (h_data[0] == 1 && h_data[N - 1] == N)
        printf("Kernel executed successfully.\n");
    else
        printf("Error in kernel execution.\n");

    // Free memory
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```