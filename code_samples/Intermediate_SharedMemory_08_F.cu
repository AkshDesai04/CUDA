/*
A different pattern is an unrolled loop: if (blockDim.x >= 512 && threadIdx.x < 256) s_data[threadIdx.x] += s_data[threadIdx.x+256]; __syncthreads(); and so on for 128, 64, etc. This can be faster.

Thinking:
The prompt describes a reduction kernel that uses an unrolled loop to sum elements in shared memory. The goal is to implement a CUDA program that demonstrates this pattern. I will create a simple vector reduction kernel that takes an input array of floats and produces a single sum. The kernel will first copy elements from global memory to shared memory and then perform a hierarchical reduction using the unrolled pattern: for each power of two step, only the first half of the threads perform the addition. I will also include a host driver that allocates memory, fills it with data, launches the kernel, and retrieves the result. The program will be a single .cu file, with a multiline comment at the top containing the exact aim from the prompt and my reasoning, followed by the complete CUDA C code. The code will compile with nvcc and run on a CUDA-capable GPU.*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    __FUNCTION__, __FILE__, __LINE__,                     \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that performs a reduction using an unrolled loop pattern.
__global__ void reduce_sum(const float *g_in, float *g_out, unsigned int n)
{
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread (if available) into shared memory.
    float sum = 0.0f;
    if (idx < n)
        sum = g_in[idx];
    if (idx + blockDim.x < n)
        sum += g_in[idx + blockDim.x];
    s_data[tid] = sum;
    __syncthreads();

    // Unrolled reduction
    if (blockDim.x >= 512 && tid < 256)
        s_data[tid] += s_data[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
        s_data[tid] += s_data[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
        s_data[tid] += s_data[tid + 64];
    __syncthreads();

    if (tid < 32)
    {
        // Warp-synchronous reduction (no __syncthreads needed)
        if (blockDim.x >= 64) s_data[tid] += s_data[tid + 32];
        if (blockDim.x >= 32) s_data[tid] += s_data[tid + 16];
        if (blockDim.x >= 16) s_data[tid] += s_data[tid + 8];
        if (blockDim.x >= 8)  s_data[tid] += s_data[tid + 4];
        if (blockDim.x >= 4)  s_data[tid] += s_data[tid + 2];
        if (blockDim.x >= 2)  s_data[tid] += s_data[tid + 1];
    }

    // Write block result to global memory
    if (tid == 0)
        g_out[blockIdx.x] = s_data[0];
}

int main()
{
    const unsigned int N = 1 << 20; // 1M elements
    const unsigned int threadsPerBlock = 512;
    const unsigned int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    size_t sizeBytes = N * sizeof(float);
    float *h_in = (float *)malloc(sizeBytes);
    for (unsigned int i = 0; i < N; ++i)
        h_in[i] = 1.0f; // simple data: sum should be N

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void **)&d_in, sizeBytes));
    CHECK_CUDA(cudaMalloc((void **)&d_out, blocks * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeBytes, cudaMemcpyHostToDevice));

    // First level reduction
    reduce_sum<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());

    // If the number of blocks is large, we need to reduce the results again.
    unsigned int s = blocks;
    float *d_in_temp = d_out;
    while (s > 1)
    {
        unsigned int nextBlocks = (s + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
        reduce_sum<<<nextBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_in_temp, d_out, s);
        CHECK_CUDA(cudaGetLastError());

        d_in_temp = d_out;
        s = nextBlocks;
    }

    float h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Sum = %f (expected %f)\n", h_result, (float)N);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);

    return 0;
}
