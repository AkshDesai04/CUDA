```cuda
/*
Aim of the program:
Show a valid use of __syncthreads() inside a conditional that is uniform across the entire block,
e.g., if (blockIdx.x == 0) { ... __syncthreads(); }.

Thinking:
- In CUDA, __syncthreads() must be executed by all threads of a block, otherwise the
  program has undefined behavior.  It is safe to place it inside a conditional if
  the condition is uniform for all threads in the block.  In this example we use
  blockIdx.x == 0 as the condition, so every thread in block 0 executes the
  __syncthreads() and every thread in other blocks skips it.  This is legal because
  the condition is uniform per block.
- The kernel demonstrates this by having block 0 initialize shared memory,
  synchronize all its threads, and then write data to global memory.  Blocks
  other than block 0 do not use the shared memory or call __syncthreads().
- The host code allocates device memory, launches the kernel with two blocks
  and 256 threads per block, and copies the results back for display.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that uses __syncthreads() inside a uniform conditional
__global__ void demoKernel(int *g_out, int N)
{
    // Declare shared memory array
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Uniform condition: all threads in block 0 execute the block
    if (blockIdx.x == 0) {
        // Each thread initializes a shared memory element
        s_data[tid] = tid * 2;          // arbitrary computation
        // All threads in block 0 must reach this point before any thread in the block
        // uses s_data. This is a valid use of __syncthreads() because the condition
        // is uniform across the block.
        __syncthreads();
    }

    // After the conditional block, all threads (including those in other blocks)
    // continue execution.  Only block 0 uses s_data; other blocks skip it.
    if (gid < N) {
        // For demonstration, write a value that depends on shared memory if
        // we are in block 0; otherwise write a different value.
        if (blockIdx.x == 0) {
            g_out[gid] = s_data[tid] + gid;
        } else {
            g_out[gid] = -1;   // indicate unused in other blocks
        }
    }
}

int main(void)
{
    const int N = 512;                     // number of elements
    const int threadsPerBlock = 256;       // threads per block
    const int numBlocks = 2;               // total number of blocks

    size_t size = N * sizeof(int);
    int *h_out = (int*)malloc(size);
    int *d_out = nullptr;

    cudaMalloc((void**)&d_out, size);
    cudaMemset(d_out, 0, size);

    // Launch kernel with dynamic shared memory of size threadsPerBlock * sizeof(int)
    demoKernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_out, N);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Output values:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    free(h_out);
    cudaFree(d_out);

    return 0;
}
```