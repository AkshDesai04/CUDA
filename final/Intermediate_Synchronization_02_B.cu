```cuda
/*
Implement the two-kernel approach for finding the maximum value.
The program demonstrates how to compute the maximum of a large array on the GPU using two separate reduction kernels.
First, each block computes a local maximum over its portion of the input array and writes that to a temporary array.
Second, a second kernel iteratively reduces this array of block maxima until a single global maximum remains.
Key points:
- Shared memory is used for intra-block reduction to minimize global memory traffic.
- Two kernels allow us to separate concerns: the first kernel deals with the original input size,
  while the second kernel deals with progressively smaller arrays (the block maxima).
- The second kernel is written generically so it can be called repeatedly until only one element remains.
- The code handles cases where the input size is not an exact multiple of the block size.
- Error checking macros ensure that CUDA API calls are verified.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Kernel 1: Each block finds the maximum over its assigned chunk of the input array.
   Shared memory is used for intra-block reduction. */
__global__ void blockMaxKernel(const float *input, float *blockMax, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element or a sentinel value if out of bounds
    float val = (idx < N) ? input[idx] : -FLT_MAX;
    sdata[tid] = val;
    __syncthreads();

    // Intra-block reduction
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        blockMax[blockIdx.x] = sdata[0];
    }
}

/* Kernel 2: Generic reduction kernel that reduces an input array to a smaller output array.
   It processes two elements per thread to improve memory coalescing. */
__global__ void reduceMaxKernel(const float *input, float *output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val1 = -FLT_MAX;
    float val2 = -FLT_MAX;

    if (idx < N)        val1 = input[idx];
    if (idx + blockDim.x < N) val2 = input[idx + blockDim.x];

    sdata[tid] = fmaxf(val1, val2);
    __syncthreads();

    // Reduce within shared memory
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result of this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/* Host function to find maximum using the two-kernel approach */
float findMax(const float *h_input, int N) {
    // Device pointers
    float *d_input = nullptr;
    float *d_blockMax = nullptr;
    float *d_blockMaxReduced = nullptr;

    // Allocate device memory for input
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_input, N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine block and grid dimensions
    const int BLOCK_SIZE = 256;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate temporary arrays for block maxima
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_blockMax, numBlocks * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_blockMaxReduced, numBlocks * sizeof(float)));

    // Launch first kernel to compute block maxima
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float);
    blockMaxKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_input, d_blockMax, N);
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Iteratively reduce the block maxima array until one element remains
    float *d_in = d_blockMax;
    float *d_out = d_blockMaxReduced;
    int currSize = numBlocks;

    while (currSize > 1) {
        int threads = BLOCK_SIZE;
        int blocks = (currSize + threads * 2 - 1) / (threads * 2);
        size_t shMem = threads * sizeof(float);

        reduceMaxKernel<<<blocks, threads, shMem>>>(d_in, d_out, currSize);
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Swap input and output for next iteration
        float *tmp = d_in;
        d_in = d_out;
        d_out = tmp;
        currSize = blocks;
    }

    // Copy result back to host
    float h_max;
    CUDA_CHECK_ERROR(cudaMemcpy(&h_max, d_in, sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(d_input));
    CUDA_CHECK_ERROR(cudaFree(d_blockMax));
    CUDA_CHECK_ERROR(cudaFree(d_blockMaxReduced));

    return h_max;
}

int main() {
    const int N = 1 << 20; // 1,048,576 elements
    float *h_array = (float *)malloc(N * sizeof(float));
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Seed RNG and fill array with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_array[i] = ((float)rand() / RAND_MAX) * 1000.0f; // values between 0 and 1000
    }

    // Find maximum using the two-kernel approach
    float maxVal = findMax(h_array, N);

    printf("Maximum value in the array: %f\n", maxVal);

    free(h_array);
    return 0;
}
```