/*
Aim: Explain in a comment why there is a limit on the number of threads per block.
Explanation:
  - The number of threads per block is limited by the GPU's architectural constraints.
  - Each Streaming Multiprocessor (SM) has a finite number of resources:
      * registers
      * shared memory
      * warp schedulers
      * L1 cache and shared memory
  - A thread block must fit into the SM's resource budgets; if it requests more threads than can be accommodated given these limits, the launch will fail or be limited by the maximum block size.
  - The maximum threads per block is also dictated by the number of warp schedulers and the ability to maintain occupancy without starving other blocks. Typical maximums are 512 or 1024 threads per block, depending on compute capability.
  - In addition, the kernel launch configuration must not exceed the hardware's maximum block dimension limits (e.g., 1024 threads per block for compute capability 3.x and newer).
  - Hence, the limit ensures that each block can be scheduled and executed efficiently on the SM without exceeding available resources.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that prints the global thread index
__global__ void printThreadIdx() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d\n", idx);
}

int main(void) {
    // Define number of threads per block (within limit)
    int threadsPerBlock = 256;  // choose a value below the typical limit of 1024
    int blocksPerGrid = 4;

    // Launch kernel
    printThreadIdx<<<blocksPerGrid, threadsPerBlock>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
