/*
Modify the kernel so that only threads in the first half of the block (i.e., `threadIdx.x < blockDim.x / 2`) print their ID.

Thought process:
1. The requirement is to change the kernel so that only half of the threads in a block perform the print.
2. In CUDA, `threadIdx.x` gives the thread's index within the block, and `blockDim.x` is the total number of threads per block.
3. Using the condition `threadIdx.x < blockDim.x / 2` will ensure only the first half of the threads (i.e., those with indices less than half of the block size) execute the print statement.
4. The kernel will use `printf` for device printing; after kernel launch we need to call `cudaDeviceSynchronize()` to ensure all output is flushed before the program exits.
5. The host code will set up a simple launch configuration (e.g., 1 block with 8 threads) to demonstrate the behavior.
6. Include minimal error checking to keep the example clear and focused on the requested modification.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints thread IDs only for the first half of the block
__global__ void printFirstHalfThreads() {
    // Only threads with index less than half the block size print
    if (threadIdx.x < blockDim.x / 2) {
        printf("Thread %d in block %d\n", threadIdx.x, blockIdx.x);
    }
}

int main() {
    // Define block and grid dimensions
    const int threadsPerBlock = 8;  // Example block size
    const int numberOfBlocks = 2;   // Example grid size

    // Launch the kernel
    printFirstHalfThreads<<<numberOfBlocks, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
