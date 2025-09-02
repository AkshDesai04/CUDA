```cpp
/*
Can threads in different blocks synchronize with `__syncthreads()`? Explain why not.

Thinking:
- `__syncthreads()` is a barrier that synchronizes all threads within the *same* block.
- It only guarantees that every thread in the block has reached the barrier before any
  thread in that block continues execution.
- The CUDA programming model does not provide any mechanism for a block to wait for
  other blocks to reach a point in the kernel; blocks execute independently.
- Therefore `__syncthreads()` cannot be used to synchronize threads in different blocks.
- To coordinate between blocks you must use higher‑level primitives such as
  `cudaDeviceSynchronize()` on the host side or atomic operations on global memory.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* Kernel that demonstrates block‑level synchronization only.
   Each block writes its block index to global memory after
   performing two __syncthreads() calls.  No cross‑block
   synchronization is attempted. */
__global__ void blockSyncKernel(int *data)
{
    /* Shared memory for the block */
    __shared__ int shared[1];
    int tid = threadIdx.x;

    /* First stage: initialize shared memory */
    if (tid == 0) shared[0] = 0;
    __syncthreads();          // Block‑level barrier

    /* Second stage: each block writes its own blockIdx.x to shared memory */
    if (tid == 0) shared[0] = blockIdx.x;
    __syncthreads();          // Block‑level barrier

    /* Write result to global memory */
    if (tid == 0) data[blockIdx.x] = shared[0];
}

int main(void)
{
    const int nBlocks = 4;
    const int threadsPerBlock = 1;

    int *devData;
    int hostData[nBlocks];

    /* Allocate device memory */
    cudaMalloc((void**)&devData, nBlocks * sizeof(int));

    /* Launch kernel with multiple blocks */
    blockSyncKernel<<<nBlocks, threadsPerBlock>>>(devData);

    /* Synchronize device to ensure kernel completion */
    cudaDeviceSynchronize();

    /* Copy results back to host */
    cudaMemcpy(hostData, devData, nBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    /* Print results */
    for (int i = 0; i < nBlocks; ++i) {
        printf("Block %d wrote %d to global memory.\n", i, hostData[i]);
    }

    /* Clean up */
    cudaFree(devData);

    return 0;
}
```