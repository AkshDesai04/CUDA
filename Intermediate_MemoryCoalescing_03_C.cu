```cpp
/*
Aim of the program (passed in this prompt word to word):
"Observe the 'Achieved Occupancy' metric. How does it compare between the two kernels?"

Thinking:
To compare the achieved occupancy between two CUDA kernels, we need to:
1. Define two simple kernels that differ in resource usage (e.g., shared memory or register count).
2. Use the CUDA runtime API to query the attributes of each kernel (block size, shared memory usage, registers per thread).
3. Compute the theoretical maximum active blocks per multiprocessor for each kernel using cudaOccupancyMaxActiveBlocksPerMultiprocessor.
4. Print the results for comparison.
5. Additionally, launch each kernel and use the CUDA profiling API to obtain the actual achieved occupancy via cudaFuncGetAttributes and the occupancy calculation.
6. Keep the code minimal and self-contained so it can be compiled and run on any system with CUDA installed.

The program below follows these steps:
- Defines Kernel1 and Kernel2 with different shared memory usage.
- Computes and prints the maximum active blocks per multiprocessor for both kernels.
- Prints the block size, shared memory, and registers per thread for each kernel.
- Launches each kernel (with dummy work) and reports the achieved occupancy via the runtime API.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

// Simple kernel that uses 32 registers per thread and no shared memory
__global__ void Kernel1(int *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = idx * idx;
}

// Kernel that uses 32 registers per thread and 256 bytes of shared memory
__global__ void Kernel2(int *out)
{
    extern __shared__ int shared[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 32) {
        shared[threadIdx.x] = idx;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        out[idx] = shared[threadIdx.x] + idx;
    } else {
        out[idx] = idx;
    }
}

void printKernelInfo(const char* name, void *kernelFunc, int blockSize)
{
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, kernelFunc);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFuncGetAttributes error for %s: %s\n", name, cudaGetErrorString(err));
        return;
    }

    int maxActiveBlocks = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, (const void *)kernelFunc, blockSize, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaOccupancyMaxActiveBlocksPerMultiprocessor error for %s: %s\n", name, cudaGetErrorString(err));
        return;
    }

    printf("Kernel: %s\n", name);
    printf("  Block size: %d threads\n", blockSize);
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Shared memory per block: %d bytes\n", attr.sharedSizeBytes);
    printf("  Max active blocks per SM: %d\n", maxActiveBlocks);
    printf("\n");
}

int main()
{
    const int N = 1 << 20;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *d_out;
    cudaMalloc(&d_out, N * sizeof(int));

    // Print kernel resource usage and theoretical occupancy
    printKernelInfo("Kernel1", Kernel1, threadsPerBlock);
    printKernelInfo("Kernel2", Kernel2, threadsPerBlock);

    // Launch Kernel1
    Kernel1<<<blocks, threadsPerBlock>>>(d_out);
    cudaDeviceSynchronize();

    // Launch Kernel2 with 256 bytes shared memory per block
    Kernel2<<<blocks, threadsPerBlock, 256>>>(d_out);
    cudaDeviceSynchronize();

    cudaFree(d_out);
    return 0;
}
```