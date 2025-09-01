```cuda
/*
Aim of the program (word to word from the prompt):
Test with N = 255, N = 256, and N = 257 with a block size of 256. Verify correctness for all cases.

My entire thinking:
The prompt only specifies that we should test with three specific array sizes and a block size of 256, and that we should verify correctness for all cases. It does not specify the actual computation that should be performed on the array. To make the test meaningful, I chose to implement a simple parallel reduction kernel that sums all elements of an integer array. This is a common CUDA pattern and allows us to test the kernel on edge cases where the array size is just below, equal to, and just above the block size.

Design decisions:
1. Use a two-pass reduction: first kernel (`reduceKernel`) computes partial sums for each block and writes them to a `blockSums` array. The second kernel (`finalReduceKernel`) sums those partial sums to produce the final result.
2. For array sizes up to one block (N <= 256), the first kernel directly writes the sum to `blockSums[0]` and we skip the second kernel.
3. The kernel uses shared memory for intra-block reduction and handles arbitrary thread counts per block (blockSize is 256).
4. CPU verification is done by a simple loop that sums the array; the result is compared against the GPU result.
5. The program runs three tests: N=255, N=256, N=257. After each test it prints whether the GPU result matches the CPU result.
6. Basic CUDA error checking is added via the `CHECK` macro.

The code is self-contained, compiles as a .cu file, and follows the instruction to output only the .cu file content. No additional files or explanations are included.

*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call or kernel launch
#define CHECK(call)                                                 \
    {                                                                \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

// First kernel: each block computes a partial sum of its portion of the array
__global__ void reduceKernel(const int *input, int *blockSums, int N)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory, zero out if out of bounds
    sdata[tid] = (idx < N) ? input[idx] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // The first thread writes the block's partial sum
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

// Second kernel: reduce the partial sums to a single value
__global__ void finalReduceKernel(int *blockSums, int numBlocks)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;

    // Load block sums into shared memory
    sdata[tid] = (tid < numBlocks) ? blockSums[tid] : 0;
    __syncthreads();

    // Reduce in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        blockSums[0] = sdata[0];
    }
}

// CPU reference sum
int cpuSum(const int *data, int N)
{
    int sum = 0;
    for (int i = 0; i < N; ++i) sum += data[i];
    return sum;
}

int main()
{
    const int blockSize = 256;
    const int testSizes[] = {255, 256, 257};
    const int numTests = sizeof(testSizes) / sizeof(testSizes[0]);

    for (int t = 0; t < numTests; ++t) {
        int N = testSizes[t];
        printf("Testing N = %d\n", N);

        // Allocate host memory and initialize
        int *h_data = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; ++i) h_data[i] = i + 1; // sum will be N*(N+1)/2

        // Allocate device memory
        int *d_data = nullptr;
        int *d_blockSums = nullptr;
        CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));

        int numBlocks = (N + blockSize - 1) / blockSize;
        CHECK(cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int)));

        // Copy data to device
        CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

        // Launch first reduction kernel
        size_t sharedMemSize = blockSize * sizeof(int);
        reduceKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_data, d_blockSums, N);
        CHECK(cudaGetLastError());

        // If more than one block, perform final reduction
        if (numBlocks > 1) {
            // Use a single block for final reduction
            int finalBlockSize = (numBlocks + blockSize - 1) / blockSize * blockSize; // ensure enough threads
            if (finalBlockSize > blockSize) finalBlockSize = blockSize;
            finalReduceKernel<<<1, finalBlockSize, finalBlockSize * sizeof(int)>>>(d_blockSums, numBlocks);
            CHECK(cudaGetLastError());
        }

        // Copy result back to host
        int gpuSumResult = 0;
        CHECK(cudaMemcpy(&gpuSumResult, d_blockSums, sizeof(int), cudaMemcpyDeviceToHost));

        // CPU reference
        int cpuSumResult = cpuSum(h_data, N);

        // Verify
        if (gpuSumResult == cpuSumResult) {
            printf("PASS: GPU sum = %d, CPU sum = %d\n", gpuSumResult, cpuSumResult);
        } else {
            printf("FAIL: GPU sum = %d, CPU sum = %d\n", gpuSumResult, cpuSumResult);
        }

        // Clean up
        free(h_data);
        CHECK(cudaFree(d_data));
        CHECK(cudaFree(d_blockSums));

        printf("\n");
    }

    return 0;
}
```