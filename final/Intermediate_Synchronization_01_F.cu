/*
Implement a version where the first kernel uses a grid-stride loop, so each thread processes multiple input elements before the block-level reduction begins.

To achieve this, the reduction kernel first performs a grid‑stride loop where each thread aggregates a chunk of the input array into a local sum. This allows each thread to touch many elements of the input before any shared memory is used. After this aggregation, the thread writes its partial sum into shared memory. The standard block‑level reduction then collapses these partial sums into a single value per block, which is written to an intermediate output array. A simple host loop repeatedly launches this kernel on the intermediate array until only one element remains, which is the final sum. The code includes error checking, memory allocation, data initialization, and cleanup. The program sums a large array of floating‑point numbers and prints the result. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void reduceKernel(const float *input, float *output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread accumulates a sum over many elements using a grid‑stride loop
    float sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Standard block‑level reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread writes the block’s partial sum to the output array
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 1 << 26; // about 67 million elements (~268 MB)
    const int threadsPerBlock = 256;
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in = (float *)malloc(bytes);
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize input array with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_in[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *d_temp1 = nullptr;
    float *d_temp2 = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));   // Will hold intermediate results
    CHECK_CUDA(cudaMalloc((void **)&d_temp1, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_temp2, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // First reduction pass
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    reduceKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_in, d_temp1, N);
    CHECK_CUDA(cudaGetLastError());

    // Iteratively reduce until a single element remains
    int srcIdx = 1; // 1 -> d_temp1, 2 -> d_temp2
    int dstIdx = 2;
    int currentSize = numBlocks;
    while (currentSize > 1) {
        int blocks = (currentSize + threadsPerBlock - 1) / threadsPerBlock;
        float *src = (srcIdx == 1) ? d_temp1 : d_temp2;
        float *dst = (dstIdx == 1) ? d_temp1 : d_temp2;

        reduceKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(src, dst, currentSize);
        CHECK_CUDA(cudaGetLastError());

        currentSize = blocks;
        // Swap src/dst for next iteration
        int temp = srcIdx;
        srcIdx = dstIdx;
        dstIdx = temp;
    }

    // Copy final result back to host
    float h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, (srcIdx == 1) ? d_temp1 : d_temp2,
                          sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference sum on CPU for verification
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_in[i];
    }

    printf("GPU sum   : %f\n", h_result);
    printf("CPU sum   : %f\n", (float)cpu_sum);
    printf("Difference: %e\n", fabs(h_result - (float)cpu_sum));

    // Cleanup
    free(h_in);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_temp1));
    CHECK_CUDA(cudaFree(d_temp2));

    return EXIT_SUCCESS;
}
