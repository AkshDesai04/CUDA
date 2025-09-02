/*
Aim: Explain 'arithmetic intensity' (ratio of math operations to memory operations). How does fusing kernels improve it?

Thinking:
The goal is to illustrate arithmetic intensity (AI), defined as the ratio of floating‑point operations (FLOPs) to the amount of memory traffic (bytes) incurred during a computation. A high AI means that a kernel performs many operations per byte transferred, which typically leads to compute‑bound behaviour; a low AI indicates memory‑bound behaviour.

To demonstrate this, we implement a simple workload: add two vectors (A + B) and then multiply the result by a scalar (C = (A + B) * alpha). We will perform this work in two ways:
1. Separate kernels – one kernel for addition, one for scaling.
2. A single fused kernel that performs both operations in one pass.

We will compute the AI for both approaches using:
AI = (#FLOPs) / (#bytes transferred).

For the separate kernels:
- Add kernel: 1 FLOP per element, reads A and B once each, writes C once → 3 memory accesses per element.
- Scale kernel: 1 FLOP per element, reads C once, writes C once → 2 memory accesses per element.
Total per element: 2 FLOPs, 5 memory accesses → AI = 2 / (5 * sizeof(float)).

For the fused kernel:
- Reads A and B once each, writes result once → 3 memory accesses per element.
- Performs 2 FLOPs per element → AI = 2 / (3 * sizeof(float)).

Thus AI increases from 2/5 to 2/3, showing improved computational density and potential performance gain.

The code below implements both approaches, measures execution time, and prints the computed AI values.
*/

#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

#define N (1 << 24)          // 16 million elements
#define THREADS_PER_BLOCK 256

// Kernel to add two vectors: C = A + B
__global__ void addKernel(const float *A, const float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Kernel to scale a vector: C = C * alpha
__global__ void scaleKernel(float *C, float alpha, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] *= alpha;
}

// Fused kernel: C = (A + B) * alpha
__global__ void fusedKernel(const float *A, const float *B, float *C, float alpha, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = (A[idx] + B[idx]) * alpha;
}

// Helper to check CUDA errors
void checkCudaErr(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    size_t bytes = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    // Initialize host data
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    checkCudaErr(cudaMalloc((void**)&d_A, bytes), "cudaMalloc d_A");
    checkCudaErr(cudaMalloc((void**)&d_B, bytes), "cudaMalloc d_B");
    checkCudaErr(cudaMalloc((void**)&d_C, bytes), "cudaMalloc d_C");

    checkCudaErr(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "Memcpy A");
    checkCudaErr(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "Memcpy B");

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float alpha = 3.0f;

    // Separate kernels
    auto start_sep = std::chrono::high_resolution_clock::now();
    addKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    checkCudaErr(cudaGetLastError(), "addKernel launch");
    scaleKernel<<<blocks, THREADS_PER_BLOCK>>>(d_C, alpha, N);
    checkCudaErr(cudaGetLastError(), "scaleKernel launch");
    checkCudaErr(cudaDeviceSynchronize(), "Device sync after separate kernels");
    auto end_sep = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_sep = end_sep - start_sep;

    // Fused kernel
    // Reset d_C to zero
    checkCudaErr(cudaMemset(d_C, 0, bytes), "cudaMemset d_C");

    auto start_fused = std::chrono::high_resolution_clock::now();
    fusedKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, alpha, N);
    checkCudaErr(cudaGetLastError(), "fusedKernel launch");
    checkCudaErr(cudaDeviceSynchronize(), "Device sync after fused kernel");
    auto end_fused = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_fused = end_fused - start_fused;

    // Copy result back to host for verification (optional)
    checkCudaErr(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "Memcpy C");

    // Compute arithmetic intensity
    // Separate: 2 FLOPs per element, 5 memory accesses per element (3 reads/writes + 2 reads/writes)
    double ai_sep = 2.0 / (5.0 * sizeof(float));
    // Fused: 2 FLOPs per element, 3 memory accesses per element
    double ai_fused = 2.0 / (3.0 * sizeof(float));

    printf("Separate kernels: time = %.3f ms, AI = %.4f\n", elapsed_sep.count(), ai_sep);
    printf("Fused kernel   : time = %.3f ms, AI = %.4f\n", elapsed_fused.count(), ai_fused);

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
