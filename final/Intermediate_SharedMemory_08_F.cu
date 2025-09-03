```cpp
/*
Aim: A different pattern is an unrolled loop: `if (blockDim.x >= 512 && threadIdx.x < 256) s_data[threadIdx.x] += s_data[threadIdx.x+256]; __syncthreads();` and so on for 128, 64, etc. This can be faster.

Thinking:
- The goal is to illustrate how an unrolled reduction loop in CUDA can outperform a standard iterative loop by reducing the number of synchronizations and leveraging warp‑level execution.
- We will implement a simple sum reduction on an array of floats.
- The kernel will load data into shared memory, perform the reduction using an unrolled pattern, and write the per‑block partial sum to an output array.
- The host will launch enough blocks so that each block processes a chunk of the input, then perform a second kernel launch (or a simple CPU sum) to combine the partial sums into the final result.
- Error checking macros will help catch any launch failures.
- The code is self‑contained and can be compiled with `nvcc sum_unrolled.cu -o sum_unrolled`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERR(call)                                   \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",  \
                    #call, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// Kernel that reduces an array using an unrolled loop
__global__ void reduce_sum_unrolled(const float *input, float *blockSums, int N)
{
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread (if within bounds)
    float mySum = 0.0f;
    if (idx < N)
        mySum = input[idx];
    if (idx + blockDim.x < N)
        mySum += input[idx + blockDim.x];

    s_data[tid] = mySum;
    __syncthreads();

    // Reduction using an unrolled loop
    if (blockDim.x >= 512) { if (tid < 256) s_data[tid] += s_data[tid + 256]; __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) s_data[tid] += s_data[tid + 128]; __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) s_data[tid] += s_data[tid +  64]; __syncthreads(); }

    // Final warp-level reduction (no __syncthreads needed)
    if (tid < 32) {
        volatile float *vsmem = s_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // Write result of this block to global memory
    if (tid == 0)
        blockSums[blockIdx.x] = s_data[0];
}

// Simple CPU sum for verification
float cpu_sum(const float *data, int N)
{
    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += data[i];
    return sum;
}

int main(int argc, char *argv[])
{
    const int N = 1 << 24; // 16M elements
    const int threadsPerBlock = 512;
    const int sharedMemBytes = threadsPerBlock * sizeof(float);

    // Allocate host memory
    float *h_in = (float *)malloc(N * sizeof(float));
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input
    for (int i = 0; i < N; ++i)
        h_in[i] = 1.0f; // Simple data for easy verification

    // Allocate device memory
    float *d_in, *d_blockSums;
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_in, N * sizeof(float)));

    // Number of blocks needed
    int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_blockSums, blocks * sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERR(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    reduce_sum_unrolled<<<blocks, threadsPerBlock, sharedMemBytes>>>(d_in, d_blockSums, N);
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    // If more than one block, reduce the partial sums on CPU
    float *h_partial = (float *)malloc(blocks * sizeof(float));
    if (!h_partial) {
        fprintf(stderr, "Failed to allocate host memory for partial sums.\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA_ERR(cudaMemcpy(h_partial, d_blockSums, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float gpu_result = 0.0f;
    for (int i = 0; i < blocks; ++i)
        gpu_result += h_partial[i];

    // Verify
    float cpu_result = cpu_sum(h_in, N);
    printf("CPU result = %f\n", cpu_result);
    printf("GPU result = %f\n", gpu_result);
    printf("Difference  = %e\n", fabs(cpu_result - gpu_result));

    // Clean up
    free(h_in);
    free(h_partial);
    CHECK_CUDA_ERR(cudaFree(d_in));
    CHECK_CUDA_ERR(cudaFree(d_blockSums));

    return 0;
}
```