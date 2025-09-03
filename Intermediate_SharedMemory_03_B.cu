```cpp
/*
Aim of the program:
Explain what would happen if you forgot the __syncthreads() after the loading phase.
(A thread might try to read a value from shared memory before another thread has written it).

Thinking behind this implementation:
1. We want to illustrate the race condition that arises when threads read from shared memory
   before all the required writes have completed. The simplest demonstration is a kernel that
   loads values from global memory into shared memory and then attempts to compute a sum
   using those shared values. If __syncthreads() is omitted after the load, the summation
   may use uninitialized entries, producing incorrect results.
2. To clearly show the difference, we implement two kernels:
   - sumKernelNoSync: omits the synchronization after loading.
   - sumKernelSync: includes the synchronization.
3. In the host code we:
   - Allocate an input array of known sequential values (0..N-1).
   - Allocate output arrays for both kernels.
   - Launch both kernels with identical configurations.
   - Copy the results back and print the first few elements of each to highlight the error.
4. The output will show that the unsynchronised kernel yields wrong partial sums, whereas
   the synchronised kernel gives correct results. This demonstrates the necessity of
   __syncthreads() when shared memory is used across threads.
5. The code is written in C++ for CUDA and compiles with nvcc. It contains basic error
   checking for clarity but keeps the example focused on the race condition demonstration.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N 64          // total number of elements
#define BLOCK_SIZE 32 // number of threads per block

// Kernel without __syncthreads() after loading
__global__ void sumKernelNoSync(const int *a, int *result, int N)
{
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N)
        shared[tid] = a[idx];  // load into shared memory

    // No synchronization here: threads may start summing before all writes are done

    // Compute partial sum of the block's shared array
    int s = 0;
    for (int j = 0; j < blockDim.x; ++j)
        s += shared[j];

    if (idx < N)
        result[idx] = s;
}

// Kernel with __syncthreads() after loading
__global__ void sumKernelSync(const int *a, int *result, int N)
{
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N)
        shared[tid] = a[idx];  // load into shared memory

    __syncthreads(); // ensure all writes are complete before reading

    // Compute partial sum of the block's shared array
    int s = 0;
    for (int j = 0; j < blockDim.x; ++j)
        s += shared[j];

    if (idx < N)
        result[idx] = s;
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void)
{
    // Allocate host memory
    int h_a[N];
    int h_resNoSync[N];
    int h_resSync[N];

    // Initialize input array with sequential values
    for (int i = 0; i < N; ++i)
        h_a[i] = i;

    // Allocate device memory
    int *d_a, *d_resNoSync, *d_resSync;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_resNoSync, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_resSync, N * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    // Determine grid dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel without synchronization
    sumKernelNoSync<<<gridDim, blockDim, BLOCK_SIZE * sizeof(int)>>>(d_a, d_resNoSync, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch kernel with synchronization
    sumKernelSync<<<gridDim, blockDim, BLOCK_SIZE * sizeof(int)>>>(d_a, d_resSync, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_resNoSync, d_resNoSync, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_resSync, d_resSync, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print first few results to illustrate the difference
    printf("Index | No Sync Result | Sync Result | Expected Partial Sum\n");
    printf("-------------------------------------------------------------\n");
    for (int i = 0; i < N; ++i)
    {
        // Expected sum for block 0: sum of first 32 elements (0..31)
        // For block 1: sum of next 32 elements (32..63)
        int blockIdx = i / BLOCK_SIZE;
        int expectedSum = 0;
        for (int j = 0; j < BLOCK_SIZE; ++j)
            expectedSum += blockIdx * BLOCK_SIZE + j;

        printf("%5d | %13d | %11d | %20d\n", i, h_resNoSync[i], h_resSync[i], expectedSum);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_resNoSync));
    CUDA_CHECK(cudaFree(d_resSync));

    return 0;
}
```