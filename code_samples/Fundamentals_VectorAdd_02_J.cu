/*
Run the program with `nvprof` or `nsight compute` to profile its performance.

This CUDA program performs a simple vector addition to give us a workload that can be
profiled.  The aim is to illustrate how to compile and run the binary under the
CUDA profiling tools.  In the comment below I describe my design choices and
thought process.

Design:
- Two input vectors (a and b) of size N are allocated on the host.
- The vectors are copied to the device, the kernel `vecAdd` is launched,
  and the result is copied back.
- The kernel uses a simple one-dimensional grid/block layout.
- N is chosen large enough (e.g., 1<<20) to produce a noticeable kernel
  execution time while remaining trivial to run on almost any GPU.

Why this program is good for profiling:
- It contains a single kernel launch, so the profiling output will focus on
  the kernel launch overhead, memory transfer times, and GPU execution time.
- The kernel is memory-bound, so you can observe the impact of memory bandwidth
  and occupancy.
- By compiling with `-O3` we enable optimizations that the profiler will
  report on (e.g., register usage, instruction throughput).

Compilation and execution:
  nvcc -O3 -o vector_add vector_add.cu

Profiling with nvprof:
  nvprof --print-gpu-trace ./vector_add

Profiling with Nsight Compute:
  nsight-cu-cli --export-profile vector_add.prof ./vector_add

Both tools will show the kernel name, launch parameters, launch latency,
kernel execution time, and memory transfer details.
*/

// CUDA vector addition example
#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)   // 1M elements

__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    size_t size = N * sizeof(float);
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Simple verification
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != 3.0f) {
            success = false;
            printf("Mismatch at %d: %f\n", i, h_c[i]);
            break;
        }
    }
    if (success) {
        printf("Vector addition successful.\n");
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
