```cpp
/********************************************************************
 * Aim of the program: Run `cuda-memcheck` with the `--tool synccheck` option to look for synchronization errors.
 *
 * My thinking:
 *   - The purpose of this CUDA program is to intentionally introduce a synchronization
 *     error so that when the program is run under `cuda-memcheck --tool synccheck` the tool
 *     will detect and report the issue.  
 *   - I chose a simple parallel reduction kernel that uses shared memory.  The
 *     kernel reads one element per thread into shared memory, then attempts to
 *     perform a tree‑style reduction to compute the sum of the block.
 *   - The crucial mistake is that I omitted a `__syncthreads()` call between the
 *     load of data into shared memory and the first reduction step.  This means
 *     that threads may start reading from `sdata` before other threads have written
 *     their values, leading to a race condition and an incorrect sum.
 *   - The host code allocates a small array, initializes it, copies it to the
 *     device, launches the kernel, and copies back the results.  The results
 *     are printed to standard output.  The program is intentionally simple
 *     so that the focus remains on the synchronization bug.
 ********************************************************************/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

// Kernel that computes the sum of an array per block using shared memory.
// Intentionally missing __syncthreads() after loading data into shared memory
// to create a synchronization error that cuda-memcheck synccheck will detect.
__global__ void misSyncKernel(const float* __restrict__ d_in,
                              float* __restrict__ d_out,
                              int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element into shared memory (may be out of bounds)
    sdata[tid] = (idx < n) ? d_in[idx] : 0.0f;

    // Intentionally omitted __syncthreads() here:
    // __syncthreads();  // This would be correct.

    // Reduce within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // No synchronization after each reduction step
    }

    // Write block result to output
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

int main()
{
    const int N = 1024;            // Number of elements
    const int BLOCK_SIZE = 256;    // Threads per block
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Host allocation
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_partial = (float*)malloc(GRID_SIZE * sizeof(float));

    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;  // Simple value to sum
    }

    // Device allocation
    float *d_in = nullptr;
    float *d_partial = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_partial, GRID_SIZE * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with shared memory size equal to BLOCK_SIZE floats
    misSyncKernel<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_in, d_partial, N);

    // Check for launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy partial sums back to host
    CHECK_CUDA(cudaMemcpy(h_partial, d_partial,
                          GRID_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Compute final sum on host
    float total = 0.0f;
    for (int i = 0; i < GRID_SIZE; ++i) {
        total += h_partial[i];
    }

    printf("Computed sum: %f (expected %f)\n", total, (float)N);

    // Clean up
    free(h_in);
    free(h_partial);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_partial));

    return 0;
}
```