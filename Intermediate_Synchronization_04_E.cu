```cuda
/*
Aim of the program:
Are atomics supported on shared memory? (Yes). Write a reduction that uses atomics on a shared memory location instead of the synchronized strided reduction.

Thinking:
1. Atomic operations can be performed on shared memory on devices of compute capability 2.x and higher.
2. The classic strided reduction uses __syncthreads() to coordinate threads in a block. 
   Here we will replace that synchronization pattern with an atomicAdd on a single
   shared memory location. Each thread will load its element and add it to the
   shared accumulator atomically. After all threads have finished, the first thread
   writes the block result to global memory.
3. Because atomicAdd on shared memory is a collective operation among the threads
   of a block, we only need a single __syncthreads() before the final write to
   ensure that all pending atomicAdd operations have completed.
4. We will launch the kernel with multiple blocks; each block computes a partial sum
   using the shared memory atomics. The host will then sum these partial sums
   (either on the CPU or with a second kernel). This demonstrates that atomics
   can be used for reductions without explicit thread-level synchronization
   beyond the final write.
5. The code includes simple error checking and a CPU reference implementation
   for correctness verification.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel that reduces an array of ints using an atomic add on shared memory.
__global__ void reduction_atomic_shared(const int *input, int n, int *blockSums)
{
    // One shared memory location per block to hold the accumulator.
    extern __shared__ int s[];
    int *sum = &s[0];

    // Initialize the shared accumulator to zero.
    if (threadIdx.x == 0) *sum = 0;
    __syncthreads();  // Ensure all threads see sum == 0.

    // Compute global index for this thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Atomically add the thread's element to the shared accumulator.
        atomicAdd(sum, input[idx]);
    }

    __syncthreads();  // Ensure all atomicAdds have completed.

    // Write the block's partial sum to global memory.
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = *sum;
    }
}

// Host-side reference implementation for correctness checking.
int cpu_reduce(const int *data, int n)
{
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += data[i];
    return sum;
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate and initialize host array.
    int *h_input = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) h_input[i] = 1; // Simple test pattern: all ones.

    // Allocate device memory.
    int *d_input, *d_blockSums;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_blockSums, blocks * sizeof(int)));

    // Copy input to device.
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with shared memory size equal to sizeof(int).
    reduction_atomic_shared<<<blocks, threadsPerBlock, sizeof(int)>>>(
        d_input, N, d_blockSums);
    CHECK_CUDA(cudaGetLastError());

    // Copy block sums back to host.
    int *h_blockSums = (int*)malloc(blocks * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_blockSums, d_blockSums, blocks * sizeof(int), cudaMemcpyDeviceToHost));

    // Final reduction on host.
    int gpuSum = 0;
    for (int i = 0; i < blocks; ++i) gpuSum += h_blockSums[i];

    // Reference sum.
    int cpuSum = cpu_reduce(h_input, N);

    printf("GPU sum (atomic shared): %d\n", gpuSum);
    printf("CPU sum          : %d\n", cpuSum);
    printf("Result is %s\n", (gpuSum == cpuSum) ? "CORRECT" : "INCORRECT");

    // Cleanup.
    free(h_input);
    free(h_blockSums);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_blockSums));

    return 0;
}
```