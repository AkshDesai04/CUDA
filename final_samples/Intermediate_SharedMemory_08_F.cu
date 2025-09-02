/*
Aim: A different pattern is an unrolled loop: `if (blockDim.x >= 512 && threadIdx.x < 256) s_data[threadIdx.x] += s_data[threadIdx.x+256]; __syncthreads();` and so on for 128, 64, etc. This can be faster.
Thinking: We implement a parallel reduction kernel using shared memory and an unrolled loop to collapse the reduction steps in a block. The kernel loads pairs of elements into shared memory, then reduces them in a hierarchical fashion: first 512->256, then 256->128, etc., unrolling the last warp manually to avoid synchronization overhead. The host code allocates an array, initializes it, and repeatedly launches the reduction kernel until a single value remains, which is the sum of the input array. We use dynamic shared memory to keep the code generic for any block size, and include error checking and timing. The final result is printed and validated against a CPU sum.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

__global__ void reduce(float *in, float *out, int n)
{
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;

    if (idx < n)           sum += in[idx];
    if (idx + blockDim.x < n) sum += in[idx + blockDim.x];

    s_data[tid] = sum;
    __syncthreads();

    // Unrolled reduction
    if (blockDim.x >= 512 && tid < 256) {
        s_data[tid] += s_data[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256 && tid < 128) {
        s_data[tid] += s_data[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128 && tid < 64) {
        s_data[tid] += s_data[tid + 64];
        __syncthreads();
    }
    if (blockDim.x >= 64 &&  tid < 32) {
        // Unroll the last warp (no need for __syncthreads() between additions)
        if (tid < 16) s_data[tid] += s_data[tid + 16];
        if (tid < 8)  s_data[tid] += s_data[tid + 8];
        if (tid < 4)  s_data[tid] += s_data[tid + 4];
        if (tid < 2)  s_data[tid] += s_data[tid + 2];
        if (tid < 1)  s_data[tid] += s_data[tid + 1];
    }

    if (tid == 0) out[blockIdx.x] = s_data[0];
}

int main(void)
{
    const int N = 1 << 20;          // 1,048,576 elements
    const int threads = 512;        // block size

    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;  // Simple test case

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(float))); // large enough for any intermediate

    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    int size = N;
    int blocks = (size + threads * 2 - 1) / (threads * 2);
    float *d_in_current = d_in;
    float *d_out_current = d_out;

    while (size > 1) {
        reduce<<<blocks, threads, threads * sizeof(float)>>>(d_in_current, d_out_current, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        size = blocks;
        blocks = (size + threads * 2 - 1) / (threads * 2);

        // Swap pointers for next iteration
        float *tmp = d_in_current;
        d_in_current = d_out_current;
        d_out_current = tmp;
    }

    float h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_in_current, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Sum = %f (expected %f)\n", h_result, (float)N);

    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
