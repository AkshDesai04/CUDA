```cuda
/*
Implement a parallel maximum-finding algorithm using the same two-pass reduction strategy as the sum reduction.
The aim of this program is to find the maximum value in a large array of floats by performing a two-pass reduction on the GPU.
The first pass reduces each block of threads to a single maximum value stored in a temporary array.
The second pass then reduces that temporary array (which contains one maximum per block) to a single global maximum.
The reduction uses shared memory for intra-block communication and a simple comparison-based reduction pattern.
The program also demonstrates how to launch successive kernel passes until a single result remains.
*/

// Include CUDA runtime API and standard headers
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel to perform block-wise maximum reduction
__global__ void blockMaxReduce(const float *input, float *blockMax, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread to reduce number of iterations
    float max_val = -FLT_MAX;
    if (i < n)
        max_val = input[i];
    if (i + blockDim.x < n)
        max_val = fmaxf(max_val, input[i + blockDim.x]);

    sdata[tid] = max_val;
    __syncthreads();

    // Parallel reduction within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0)
        blockMax[blockIdx.x] = sdata[0];
}

// Kernel to reduce array of block maxima to a smaller array of maxima
__global__ void reduceMax(const float *input, float *output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float max_val = -FLT_MAX;
    if (i < n)
        max_val = input[i];
    if (i + blockDim.x < n)
        max_val = fmaxf(max_val, input[i + blockDim.x]);

    sdata[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Utility function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Size of the input array (e.g., 2^20 elements)
    const int N = 1 << 20;
    const int THREADS_PER_BLOCK = 256;

    // Allocate host memory and initialize with random floats
    float *h_input = (float *)malloc(N * sizeof(float));
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i)
        h_input[i] = (float)rand() / RAND_MAX * 1000.0f; // random values between 0 and 1000

    // Allocate device memory
    float *d_input;
    float *d_blockMax;
    checkCudaError(cudaMalloc((void **)&d_input, N * sizeof(float)), "cudaMalloc d_input");

    // Copy input to device
    checkCudaError(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy to device");

    // Determine number of blocks for first pass
    int blocks = (N + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    checkCudaError(cudaMalloc((void **)&d_blockMax, blocks * sizeof(float)), "cudaMalloc d_blockMax");

    // First pass: block-wise reduction to compute per-block maxima
    size_t sharedMemSize = THREADS_PER_BLOCK * sizeof(float);
    blockMaxReduce<<<blocks, THREADS_PER_BLOCK, sharedMemSize>>>(d_input, d_blockMax, N);
    checkCudaError(cudaGetLastError(), "Kernel launch blockMaxReduce");

    // Second pass: iteratively reduce block maxima to a single value
    int currentSize = blocks;
    float *d_in = d_blockMax;
    float *d_out = NULL;
    float *d_temp = NULL;

    // We will allocate a temporary buffer for each reduction step
    while (currentSize > 1)
    {
        int nextBlocks = (currentSize + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
        if (d_temp == NULL)
            checkCudaError(cudaMalloc((void **)&d_temp, nextBlocks * sizeof(float)), "cudaMalloc d_temp");

        reduceMax<<<nextBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(d_in, d_temp, currentSize);
        checkCudaError(cudaGetLastError(), "Kernel launch reduceMax");

        // Prepare for next iteration
        if (d_in != d_blockMax)
            cudaFree(d_in);
        d_in = d_temp;
        currentSize = nextBlocks;
    }

    // Copy the final maximum back to host
    float h_max;
    checkCudaError(cudaMemcpy(&h_max, d_in, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy from device");

    // Clean up
    cudaFree(d_input);
    cudaFree(d_blockMax);
    if (d_in != d_blockMax)
        cudaFree(d_in);

    // Print result
    printf("Maximum value found: %f\n", h_max);

    free(h_input);
    return 0;
}
```