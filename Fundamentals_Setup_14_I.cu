/*
Aim of the program: Write a short paragraph in comments explaining the benefits of overlapping data transfers and computation.

Benefits of overlapping data transfers and computation:
When a CUDA program performs data transfer between host and device asynchronously (using cudaMemcpyAsync) and simultaneously launches kernels in a stream, the GPU can execute memory copies and compute operations concurrently. This overlap reduces idle time on both the CPU and GPU, improves overall throughput, and leads to better utilization of the PCI‑e bandwidth and GPU compute resources. As a result, the total execution time can be significantly lower compared to performing transfers and kernels sequentially, especially for large data sets and kernels that take non‑negligible time to execute.

Thinking process:
I wanted to illustrate overlapping by writing a small vector addition program that uses a single CUDA stream. By issuing two asynchronous copies, launching the kernel, and then copying the result back all in the same stream, the GPU can perform the copies and kernel execution concurrently. I added comments explaining why this is beneficial and included a simple verification print to show that the kernel worked. The program is self‑contained and can be compiled with nvcc to demonstrate the concept.
*/

#include <cuda_runtime.h>
#include <stdio.h>

// Simple vector addition kernel
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main()
{
    const int N = 1 << 20;           // 1 million elements
    size_t size = N * sizeof(float);

    // Host memory allocation
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = 2.0f * static_cast<float>(i);
    }

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Create a single CUDA stream for overlapping
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronous host-to-device copies
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);

    // Kernel launch (also in the same stream)
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, N);

    // Asynchronous device-to-host copy
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);

    // Wait for all operations in the stream to complete
    cudaStreamSynchronize(stream);

    // Verify a few results
    printf("c[0] = %f, c[1] = %f\n", h_c[0], h_c[1]);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaStreamDestroy(stream);

    return 0;
}
