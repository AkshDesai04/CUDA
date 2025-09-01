```c
/*
Aim of the program:
Combine the concepts of the previous two questions. Launch a kernel with a grid of 4 blocks, each containing 16 threads. Each thread must calculate and print its unique 'global ID' using the formula: `int id = blockIdx.x * blockDim.x + threadIdx.x;`.

Thinking process:
1. Use CUDA C with a single kernel that each thread executes.
2. Compute the global ID with the given formula.
3. Use device printf to output each thread's local and global IDs. Device printf requires that the GPU supports it and that the host synchronizes after launch to ensure the output is flushed.
4. Launch the kernel with a dim3 grid of (4,1,1) and a dim3 block of (16,1,1).
5. After launch, call cudaDeviceSynchronize() to wait for completion and flush the prints.
6. Handle any errors from kernel launch or synchronization for robustness.
7. No memory allocation is necessary since we only print IDs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_global_ids()
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d -> Global ID %d\n",
           blockIdx.x, threadIdx.x, id);
}

int main(void)
{
    // Launch kernel with 4 blocks, each having 16 threads
    dim3 threadsPerBlock(16, 1, 1);
    dim3 numBlocks(4, 1, 1);

    print_global_ids<<<numBlocks, threadsPerBlock>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish before accessing any device results
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel: %s\n",
                err, cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```