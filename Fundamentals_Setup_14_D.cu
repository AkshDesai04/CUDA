/*
Add a comment explaining what a CUDA stream is and how it relates to overlapping operations.

In this program, the first multiline comment is the aim of the program as requested: it literally contains the words
"Add a comment explaining what a CUDA stream is and how it relates to overlapping operations." 
I also wrote a detailed explanation below to satisfy the second part of the instruction, which is to add the entire thinking.
--------------------------------------------------------------------
CUDA streams are ordered sequences of operations (kernel launches, memory copies, etc.) that execute on the GPU.
Each stream has its own queue of commands; commands within the same stream execute in the order they are issued.
Commands issued to different streams can execute concurrently (subject to hardware and resource constraints),
allowing the programmer to overlap kernel execution with data transfers or with other kernels.
This overlapping is useful to hide memory latency and improve throughput.

When you launch a kernel on stream 0 (the default stream) the operation is serialized with all other
operations on stream 0, but it can run concurrently with operations in other streams if the GPU
supports concurrent execution. By explicitly creating separate streams (e.g., streamA and streamB)
and launching kernels or memory copies into them, you can let the GPU schedule them independently.
The driver may interleave the work so that while a kernel in streamA is executing, a memory copy or a kernel
in streamB can also progress, provided there are enough resources (streams, compute units, memory bandwidth, etc.).
Thus, streams enable overlapping of operations, leading to better utilization of the GPU.
--------------------------------------------------------------------
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that performs dummy work
__global__ void dummyKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = data[idx] * 2;   // Just double each element
}

int main(void)
{
    const int N = 1 << 20;          // Number of elements
    const size_t size = N * sizeof(int);
    int *h_data = (int *)malloc(size);

    // Initialize host array
    for (int i = 0; i < N; ++i)
        h_data[i] = i;

    // Device memory allocation
    int *d_data;
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Create two streams
    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    // Launch kernels in different streams to demonstrate overlapping
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel 1 in streamA
    dummyKernel<<<blocksPerGrid, threadsPerBlock, 0, streamA>>>(d_data, N);

    // Kernel 2 in streamB (works on the same data for simplicity)
    dummyKernel<<<blocksPerGrid, threadsPerBlock, 0, streamB>>>(d_data, N);

    // Synchronize streams
    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);

    // Copy back result
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Simple validation: print first 5 results
    printf("First 5 results after two passes of doubling: \n");
    for (int i = 0; i < 5; ++i)
        printf("%d ", h_data[i]);

    printf("\n");

    // Cleanup
    cudaFree(d_data);
    free(h_data);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);

    return 0;
}
