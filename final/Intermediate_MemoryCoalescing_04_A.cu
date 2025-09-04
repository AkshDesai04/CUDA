/*
Run for `STRIDE=32`. The performance should be very poor.

The purpose of this program is to illustrate how memory access patterns
affect the performance of a CUDA kernel.  We intentionally use a
strided memory layout where each thread reads from an element that is
separated from its neighbours by 32 floats.  This pattern is highly
uncoalesced for typical GPU architectures and therefore will yield
very poor performance.

Design decisions:
- The kernel performs a simple element‑wise addition of two large
  arrays, but each thread accesses the input arrays at an offset
  `idx * STRIDE`.  Because the stride (32) is larger than the warp
  size, neighbouring threads in a warp will load data from distant
  memory locations.  As a result, the hardware cannot coalesce the
  global memory transactions, leading to a high number of memory
  transactions and low bandwidth utilisation.
- The host code allocates arrays that are large enough to demonstrate
  the performance issue (1M elements per output array, with a stride
  of 32, meaning each input array is 32M floats).  Random data is
  generated for the input arrays.
- Timing is performed with CUDA events so that we can compare the
  execution time to that of a kernel that uses contiguous memory
  accesses (not included here).  The expected runtime will be much
  higher than a coalesced implementation.
- The program is intentionally simple and self‑contained: it
  compiles as a single .cu file and can be run on any CUDA capable
  device.

The user can adjust `N` and `STRIDE` to experiment with different
sizes, but for this demonstration `STRIDE` is set to 32 and the
performance should indeed be poor.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STRIDE 32
#define N 1000000            // number of output elements
#define THREADS_PER_BLOCK 256

// Simple element‑wise addition kernel with strided access
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Uncoalesced access: each thread jumps 32 floats
        int offset = idx * STRIDE;
        c[idx] = a[offset] + b[offset];
    }
}

// Utility to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    // Allocate host memory
    size_t size_out = N * sizeof(float);
    size_t size_in  = N * STRIDE * sizeof(float);
    float *h_a = (float*)malloc(size_in);
    float *h_b = (float*)malloc(size_in);
    float *h_c = (float*)malloc(size_out);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input arrays with random data
    for (int i = 0; i < N * STRIDE; ++i) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc((void**)&d_a, size_in), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void**)&d_b, size_in), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void**)&d_c, size_out), "cudaMalloc d_c");

    // Copy inputs to device
    checkCudaError(cudaMemcpy(d_a, h_a, size_in, cudaMemcpyHostToDevice), "cudaMemcpy h_a->d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, size_in, cudaMemcpyHostToDevice), "cudaMemcpy h_b->d_b");

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    addKernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Measure elapsed time
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");

    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c, d_c, size_out, cudaMemcpyDeviceToHost), "cudaMemcpy d_c->h_c");

    // Simple verification: compute sum of output array
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += h_c[i];
    }
    printf("Sum of result array: %f\n", sum);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
