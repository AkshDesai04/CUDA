```cuda
/*
Aim of the program:
For the final 32 elements (one warp), __syncthreads is not needed between iterations.
Use warp shuffle intrinsics (__shfl_down_sync) for a highly optimized final stage of the reduction.

Thinking:
The goal is to implement a fast parallel reduction on the GPU that takes advantage of warp shuffle
intrinsics for the last stage of the reduction.  A typical two‑step reduction is used:
1. Each block loads its portion of the input array into shared memory and performs a
   block‑level reduction.  The first part of the reduction is done with a classic
   loop that repeatedly halves the active threads and uses __syncthreads() to keep
   all threads in sync after each iteration.
2. When the active thread count drops to 32 (one full warp), the remaining reduction can
   be performed without any synchronization because all threads in a warp execute
   in lock‑step.  We load the 32 partial sums from shared memory into a register,
   then use __shfl_down_sync() to perform a warp‑level reduction.  The mask
   0xffffffff ensures that all 32 threads participate.
3. The first thread of the warp writes the block's result to an output array.
   The host then sums these block results to obtain the final sum.

This approach reduces the number of __syncthreads() calls, which can be a performance
bottleneck, especially for kernels with a large number of blocks.  The warp shuffle
intrinsics are highly efficient because they use the GPU's hardware shuffle logic
instead of shared memory or global memory traffic.

The program below demonstrates this technique on a large array of floats.
It allocates an array on the host, copies it to the device, launches the reduction
kernel, copies the per‑block results back, and finally reduces them on the host.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",       \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),           \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that performs reduction on a chunk of the input array
// Each block processes blockDim.x * 2 elements (two per thread for coalesced access)
// The partial sum for each block is written to the output array
__global__ void reduceKernel(const float *input, float *output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    // Load two elements per thread and sum them
    float sum = 0.0f;
    if (idx < n)               sum += input[idx];
    if (idx + blockDim.x < n)  sum += input[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Standard reduction in shared memory (loop halving the active threads)
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // synchronize after each stage
    }

    // Final warp reduction using __shfl_down_sync (no __syncthreads needed)
    if (tid < 32)
    {
        float val = sdata[tid];
        // Shuffle down within the warp
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        // Thread 0 writes the block's partial sum
        if (tid == 0)
            output[blockIdx.x] = val;
    }
}

// Host function to perform the full reduction by launching the kernel
float reduce(const float *h_input, int n)
{
    float *d_input = nullptr;
    float *d_blockSums = nullptr;
    float *h_blockSums = nullptr;
    int threadsPerBlock = 256; // must be >= 32
    int blocksPerGrid = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    size_t inputSize = n * sizeof(float);
    size_t blockSumsSize = blocksPerGrid * sizeof(float);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_input, inputSize));
    CHECK_CUDA(cudaMalloc((void **)&d_blockSums, blockSumsSize));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

    // Launch kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    reduceKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_blockSums, n);
    CHECK_CUDA(cudaGetLastError());

    // Copy block sums back to host
    h_blockSums = (float *)malloc(blockSumsSize);
    CHECK_CUDA(cudaMemcpy(h_blockSums, d_blockSums, blockSumsSize, cudaMemcpyDeviceToHost));

    // Final reduction on host (simple serial sum)
    float total = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i)
        total += h_blockSums[i];

    // Clean up
    free(h_blockSums);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_blockSums));

    return total;
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    float *h_input = (float *)malloc(N * sizeof(float));
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with some values (e.g., all ones)
    for (int i = 0; i < N; ++i)
        h_input[i] = 1.0f;

    // Perform reduction
    float result = reduce(h_input, N);

    printf("Reduction result: %f\n", result);
    printf("Expected result: %f\n", (float)N);

    free(h_input);
    return 0;
}
```