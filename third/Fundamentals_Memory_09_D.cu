/*
What is the downside of using pinned memory? (It's a limited resource and reduces memory available to the OS and other applications).

[Thinking: This program demonstrates the use of pinned (page-locked) host memory in CUDA.  
Pinned memory allows faster DMA transfers between host and device because the
operating system guarantees that the pages remain resident in physical memory.
However, allocating pinned memory consumes system physical RAM and the OS
cannot swap these pages out, so large allocations can reduce memory available
for other applications and increase pressure on the system.  
To illustrate, this example allocates three pinned arrays on the host, copies
them to the device, performs a simple element‑wise addition kernel, and copies
the result back.  The code includes basic error checking and prints the first
few results to verify correctness.  Running this with very large array sizes
would quickly exhaust system memory, showing the downside in practice.  */

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Simple kernel: c[i] = a[i] + b[i];
__global__ void vecAdd(const float *a, const float *b, float *c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main(void)
{
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    float *h_a, *h_b, *h_c; // pinned host memory
    float *d_a, *d_b, *d_c; // device memory

    // Allocate pinned host memory
    CUDA_CHECK(cudaHostAlloc((void**)&h_a, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_b, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_c, bytes, cudaHostAllocDefault));

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("Sample results:\n");
    for (size_t i = 0; i < 5; ++i) {
        printf("c[%zu] = %f\n", i, h_c[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));

    return 0;
}
