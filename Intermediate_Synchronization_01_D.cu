```cuda
/* 
   An alternative to a second kernel is to copy the d_partial_sums array back to the host and sum it on the CPU. 

   Thought process:
   1. We want to demonstrate how to compute the sum of a large array on the GPU in two stages:
      a) Launch a kernel that computes partial sums per block and stores them in an array d_partial_sums.
      b) Instead of launching a second kernel to accumulate those partial sums on the GPU, we copy d_partial_sums
         back to host memory and finish the accumulation on the CPU.
   2. The kernel will use shared memory to reduce the thread-block's elements to a single sum.
   3. The host will then iterate over the partial sums and compute the final sum.
   4. For simplicity, we allocate a fixed-size array on the device, initialize it on the host, and copy it over.
   5. We include basic error checking, timing, and output of the final sum to verify correctness.
   6. The code is self-contained and can be compiled with nvcc:
        nvcc -o sum_partial sum_partial.cu
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA kernel: each block computes the sum of a subset of the input array
__global__ void partialSumKernel(const float *d_in, float *d_partial_sums, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory, or zero if out of bounds
    float val = (idx < N) ? d_in[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Intra-block reduction (binary tree)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread writes the block's sum to the output array
    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main() {
    const int N = 1 << 20;          // Number of elements (about 1 million)
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float *h_in = (float *)malloc(N * sizeof(float));
    if (!h_in) { fprintf(stderr, "Failed to allocate host input array\n"); return EXIT_FAILURE; }

    // Initialize input array with some values (e.g., all ones)
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    // Allocate device memory
    float *d_in = nullptr;
    float *d_partial_sums = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_partial_sums, blocks * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel to compute partial sums
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    partialSumKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_in, d_partial_sums, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums back to host
    float *h_partial_sums = (float *)malloc(blocks * sizeof(float));
    if (!h_partial_sums) { fprintf(stderr, "Failed to allocate host partial sums array\n"); return EXIT_FAILURE; }
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Sum partial sums on CPU
    double host_sum = 0.0;
    for (int i = 0; i < blocks; ++i) {
        host_sum += h_partial_sums[i];
    }

    // Verify result
    double expected = static_cast<double>(N) * 1.0;
    printf("Computed sum = %f\n", host_sum);
    printf("Expected sum = %f\n", expected);
    if (abs(host_sum - expected) < 1e-6) {
        printf("Result is correct.\n");
    } else {
        printf("Result is incorrect!\n");
    }

    // Clean up
    free(h_in);
    free(h_partial_sums);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial_sums));

    return 0;
}
```