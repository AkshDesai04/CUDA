/*
Aim: Modify the kernel so that only threads with an even index print their ID.

Thinking:
1. The original kernel likely prints the thread ID for every thread.
2. To restrict printing to only even-indexed threads, we need a condition inside the kernel
   that checks if the computed global thread index is even (i.e., divisible by 2).
3. Use the standard CUDA thread indexing formula: 
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
4. Then apply: if (idx % 2 == 0) { printf(...); }
5. Ensure we include necessary headers for printf in device code: <stdio.h> and <cuda_runtime.h>.
6. In the host code, set up a simple launch configuration that creates several blocks and threads
   so we can see the output. We'll use a modest number (e.g., 256 threads per block, 4 blocks)
   giving 1024 total threads. Only even ones (512 threads) will print.
7. After launching, synchronize the device and exit cleanly.
8. The code must be a single .cu file, starting with the comment block and followed by the full code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printEvenThreadId(void)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Only print if the global thread index is even
    if (idx % 2 == 0) {
        printf("Thread ID %d\n", idx);
    }
}

int main(void)
{
    // Define launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = 4;  // 4 * 256 = 1024 total threads

    // Launch the kernel
    printEvenThreadId<<<blocksPerGrid, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optional: check for any kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
