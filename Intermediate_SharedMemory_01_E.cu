/*
Use `nsight compute` or `nvprof` to measure the global memory bandwidth and transaction count for the naive kernel. Note the high number of transactions.

Thinking:
The goal is to write a minimal CUDA program that launches a "naive" kernel, i.e., one that accesses global memory in a non‑coalesced way so that the GPU will issue many memory transactions.  After compiling this program with `nvcc` and running it, the user can run `nvprof` or Nsight Compute to observe the global memory bandwidth and transaction count metrics.

Implementation details:
- The kernel will read from a global array using a stride pattern that misaligns threads on memory addresses (e.g., `a[i*stride]` where `stride` is larger than the warp size).  This forces each thread to load data that is not contiguous, resulting in many independent transactions.
- The kernel simply copies data from an input array to an output array to keep the kernel simple.
- The host code allocates and initializes the arrays, copies them to the device, launches the kernel, copies the results back, and cleans up.
- Simple error checking is included for clarity.

Compile with:
    nvcc -o naive_mem naive_mem.cu

Run and profile with:
    nvprof ./naive_mem
    or
    nsight-compute ./naive_mem

Now the actual code:
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N  (1 << 20)          // 1M elements
#define BLOCK_SIZE 256

// Naive kernel: each thread reads from a large stride and writes to output
__global__ void naiveKernel(const float* in, float* out, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Stride > warp size to cause uncoalesced accesses
        int inIdx = idx * stride;
        if (inIdx < N) {
            out[idx] = in[inIdx];
        } else {
            out[idx] = 0.0f;
        }
    }
}

int main(void)
{
    // Host allocation
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    if (!h_in || !h_out) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)i;
    }

    // Device allocation
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int stride = 32;  // stride larger than warp size (32) to force uncoalesced accesses

    // Launch naive kernel
    naiveKernel<<<gridDim, blockDim>>>(d_in, d_out, stride);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output back to host (optional, not needed for profiling)
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}