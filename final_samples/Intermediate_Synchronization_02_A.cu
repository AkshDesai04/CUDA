```cuda
/*
The reduction operation in shared memory is now `s_data[i] = max(s_data[i], s_data[i+s])` instead of `+=`.

My thought process for this implementation:
1. I want to showcase how to change the classic sum reduction to a max reduction.
2. The kernel will load elements into shared memory, then iteratively compare pairs
   using `max` instead of adding them.
3. I will use two kernels:
   - `blockMaxKernel`: each block reduces its portion of the array to a single maximum
     and writes it to an auxiliary array `d_blockMax`.
   - `finalMaxKernel`: reduces the block maxima to a single global maximum.
4. I choose `int` for simplicity; using `INT_MIN` as the identity element.
5. I handle cases where the array size is not an exact multiple of the block size
   by setting out‑of‑range threads to `INT_MIN`.
6. I add basic error checking after CUDA calls.
7. The host code fills an input array with random integers, copies it to device,
   runs the kernels, copies back the final result, and prints it.
8. I keep the code self‑contained and compile‑ready.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <limits.h>
#include <time.h>

// Kernel to compute max within each block
__global__ void blockMaxKernel(const int *d_in, int *d_blockMax, int N)
{
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory (use INT_MIN for out‑of‑range indices)
    s_data[tid] = (gid < N) ? d_in[gid] : INT_MIN;

    __syncthreads();

    // Reduction within block using max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = max(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        d_blockMax[blockIdx.x] = s_data[0];
    }
}

// Kernel to reduce block maxima to a single global maximum
__global__ void finalMaxKernel(const int *d_blockMax, int *d_result, int numBlocks)
{
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load block max into shared memory (use INT_MIN for out‑of‑range indices)
    s_data[tid] = (gid < numBlocks) ? d_blockMax[gid] : INT_MIN;

    __syncthreads();

    // Reduction within block using max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = max(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }

    // Write final result to global memory
    if (tid == 0) {
        d_result[0] = s_data[0];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main(void)
{
    const int N = 1 << 20; // 1M elements
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    int *h_in = (int *)malloc(N * sizeof(int));
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with random integers
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_in[i] = rand() % 1000; // values between 0 and 999
    }

    // Allocate device memory
    int *d_in, *d_blockMax, *d_result;
    CUDA_CHECK(cudaMalloc((void **)&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_blockMax, numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_result, sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch block max kernel
    size_t sharedMemSize = threadsPerBlock * sizeof(int);
    blockMaxKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_in, d_blockMax, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch final max kernel (only one block needed)
    finalMaxKernel<<<1, threadsPerBlock, sharedMemSize>>>(d_blockMax, d_result, numBlocks);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    int h_max;
    CUDA_CHECK(cudaMemcpy(&h_max, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // Compute host reference maximum for verification
    int host_max = INT_MIN;
    for (int i = 0; i < N; ++i) {
        if (h_in[i] > host_max) host_max = h_in[i];
    }

    printf("GPU maximum: %d\n", h_max);
    printf("CPU maximum: %d\n", host_max);
    printf("Match: %s\n", (h_max == host_max) ? "YES" : "NO");

    // Clean up
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_blockMax));
    CUDA_CHECK(cudaFree(d_result));

    return EXIT_SUCCESS;
}
```