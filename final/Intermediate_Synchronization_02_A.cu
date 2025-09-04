```cuda
/*
Aim: Perform a parallel reduction to find the maximum value in an array using CUDA.
The reduction operation in shared memory is now s_data[i] = max(s_data[i], s_data[i+s]) instead of summation.

Thinking:
- The program will allocate a large array of floats on the host, initialize it with random values.
- It will copy the array to device memory.
- A first kernel (reduceMaxKernel) will launch one block per chunk of the array.
  Each block loads its chunk into shared memory, then performs a binary tree reduction
  using the max operation: s_data[i] = max(s_data[i], s_data[i+s]).
  The block writes its local maximum to a global array of block results.
- A second kernel (finalReduceMaxKernel) will reduce the block results array to a single
  maximum value, again using shared memory and the max operation.
- The final result is copied back to the host and printed.
- Error checking is performed after CUDA API calls.
- The block size is chosen as 256 threads, which is a typical size for many GPUs.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel to compute per-block maximums
__global__ void reduceMaxKernel(const float *input, float *blockMax, size_t N) {
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread into shared memory (coalesced)
    float val = (idx < N) ? input[idx] : -FLT_MAX;
    float val2 = (idx + blockDim.x < N) ? input[idx + blockDim.x] : -FLT_MAX;
    s_data[tid] = max(val, val2);

    __syncthreads();

    // Reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = max(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        blockMax[blockIdx.x] = s_data[0];
    }
}

// Kernel to reduce block maximums to a single value
__global__ void finalReduceMaxKernel(float *blockMax, size_t N) {
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;

    // Load block results into shared memory
    s_data[tid] = (tid < N) ? blockMax[tid] : -FLT_MAX;
    __syncthreads();

    // Reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = max(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        blockMax[0] = s_data[0];
    }
}

int main() {
    const size_t N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host allocation
    float *h_data = (float*)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize data with random values
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = (float)rand() / RAND_MAX * 1000.0f;
    }

    // Device allocation
    float *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Determine block and grid sizes
    const int BLOCK_SIZE = 256;
    int gridSize = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    // Allocate array for block maxima
    float *d_blockMax = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_blockMax, gridSize * sizeof(float)));

    // Launch first kernel
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float);
    reduceMaxKernel<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(d_data, d_blockMax, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch final reduction kernel
    // We may need to iterate if gridSize > BLOCK_SIZE, but here we assume gridSize <= BLOCK_SIZE
    // For generality, we use a loop
    int curSize = gridSize;
    while (curSize > 1) {
        int curGrid = (curSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        finalReduceMaxKernel<<<curGrid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_blockMax, curSize);
        CUDA_CHECK(cudaGetLastError());
        curSize = curGrid;
    }

    // Copy result back to host
    float h_max;
    CUDA_CHECK(cudaMemcpy(&h_max, d_blockMax, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Maximum value: %f\n", h_max);

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_blockMax));
    free(h_data);

    return 0;
}
```