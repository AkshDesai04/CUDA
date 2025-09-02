```cuda
/*
Look for the gld_transactions and gst_transactions metrics in the profiler. The naive version should have many more read transactions (gld).

Thinking:
  The goal is to provide a small CUDA program that demonstrates the effect of memory access patterns on the GPU profiler metrics
  gld_transactions (global load transactions) and gst_transactions (global store transactions).  A "naive" kernel will access global
  memory in a non-coalesced fashion so that consecutive threads in a warp read far apart addresses.  An "optimized" kernel will
  access consecutive addresses so that the accesses are coalesced.  The two kernels perform a simple operation (e.g. adding 1 to
  each element) and write the result back to global memory.  By running the program with Nsight Compute or nvprof, a user can
  observe that the naive kernel generates many more gld_transactions than the optimized kernel.

  Implementation details:
    * Allocate a large array on the host, initialize it, copy it to the device.
    * Launch both kernels with identical launch parameters (blocks, threads).
    * Measure execution time with cudaEvent but the focus is on profiling metrics.
    * After each kernel, copy the result back to host (optional for correctness).
    * Clean up memory.

  Kernel design:
    - Naive kernel: each thread reads from `in[idx]` where idx = blockIdx.x * blockDim.x + threadIdx.x
                      but then adds an offset of `threadIdx.x * 4` to create a stride of 4 elements.
                      Thus consecutive threads in a warp read indices separated by 4 elements (16 bytes).
                      This pattern is poorly coalesced and should generate many gld_transactions.
    - Optimized kernel: each thread reads from `in[idx]` where idx = blockIdx.x * blockDim.x + threadIdx.x
                        with no additional offset, yielding contiguous memory accesses that are coalesced.

  The program can be compiled with nvcc and profiled with `nvprof` or Nsight Compute:
      nvcc -o memory_profiler memory_profiler.cu
      nvprof --metrics gld_transactions,gst_transactions ./memory_profiler
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Number of elements (must be divisible by block size for simplicity)
const int N = 1 << 24; // 16M elements (~64MB for int)
const int BLOCK_SIZE = 256;

// Naive kernel: non-coalesced global load pattern
__global__ void naiveKernel(const int *in, int *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Introduce a stride to break coalescing: each thread reads an element offset by 4 * threadIdx.x
        // This means consecutive threads read addresses separated by 16 bytes.
        int offsetIdx = idx + threadIdx.x * 4;
        if (offsetIdx < N)
            out[idx] = in[offsetIdx] + 1;
        else
            out[idx] = in[idx] + 1; // fallback to avoid out-of-bounds
    }
}

// Optimized kernel: coalesced global load pattern
__global__ void optimizedKernel(const int *in, int *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        out[idx] = in[idx] + 1;
    }
}

// Simple error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main()
{
    // Host allocation
    int *h_in  = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host input
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    // Device allocation
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, N * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch naive kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaEventRecord(start));
    naiveKernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msNaive = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msNaive, start, stop));

    // Copy result back (optional)
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Launch optimized kernel
    CUDA_CHECK(cudaEventRecord(start));
    optimizedKernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msOptimized = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msOptimized, start, stop));

    // Copy result back (optional)
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print timing
    printf("Naive kernel time:   %f ms\n", msNaive);
    printf("Optimized kernel time: %f ms\n", msOptimized);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
```