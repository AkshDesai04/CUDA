/*
Implement a parallel sum reduction for a large array that does not fit in a single block. The first kernel launch reduces each block's segment of the array into a partial sum (using the shared memory technique). A second, smaller kernel launch then sums up these partial results.

Thought process:
1. We need a two-pass reduction. First kernel processes the input array in chunks that fit within a block, using shared memory to accumulate the sum for each block. Each block writes its partial sum to an auxiliary array.
2. The second kernel reduces the array of partial sums. Since the number of partial sums is usually much smaller, we can launch a single block or a few blocks to finish the reduction.
3. For generality, we keep the same reduction pattern in the second kernel. It also uses shared memory and a tree-based reduction.
4. We choose a block size of 256 threads (common choice). The reduction uses a warp‑level sync inside the loop to avoid unnecessary __syncthreads() after the last step.
5. Host code: allocate device memory, copy input, compute number of blocks for first kernel, launch it, then compute blocks for second kernel, launch it, copy result back, and free resources.
6. Simple error checking helper to catch CUDA runtime errors.
7. Example usage: fill input array with values 1..N, so the sum should be N*(N+1)/2, which we print for verification.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Helper macro for CUDA error checking
inline void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// First kernel: reduce each block's segment into a partial sum
__global__ void reduceKernel1(const float *input, float *partialSums, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; // stride of 2*blockDim

    // Load elements into shared memory, handling bounds
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Intra‑block reduction (tree reduction)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0) partialSums[blockIdx.x] = sdata[0];
}

// Second kernel: reduce partial sums into final result
__global__ void reduceKernel2(const float *partialSums, float *result, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) sum += partialSums[idx];
    if (idx + blockDim.x < n) sum += partialSums[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) result[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 1 << 20; // 1,048,576 elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f; // simple pattern

    // Allocate device memory
    float *d_input = nullptr;
    float *d_partial = nullptr;
    float *d_result = nullptr;
    checkCuda(cudaMalloc((void**)&d_input, size), "cudaMalloc d_input");
    checkCuda(cudaMalloc((void**)&d_partial, size), "cudaMalloc d_partial"); // max size
    checkCuda(cudaMalloc((void**)&d_result, sizeof(float)), "cudaMalloc d_result");

    // Copy input to device
    checkCuda(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // First kernel launch parameters
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    reduceKernel1<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_partial, N);
    checkCuda(cudaGetLastError(), "reduceKernel1 launch");

    // Second kernel launch parameters
    int blocksPerGrid2 = (blocksPerGrid + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    reduceKernel2<<<blocksPerGrid2, threadsPerBlock, sharedMemSize>>>(d_partial, d_result, blocksPerGrid);
    checkCuda(cudaGetLastError(), "reduceKernel2 launch");

    // Copy result back to host
    float h_result = 0.0f;
    checkCuda(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    printf("Parallel reduction result: %f\n", h_result);
    printf("Expected result: %f\n", (float)N * 1.0f); // since each element is 1.0f

    // Clean up
    cudaFree(d_input);
    cudaFree(d_partial);
    cudaFree(d_result);
    free(h_input);

    return 0;
}
