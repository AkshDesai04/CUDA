/*
Place a `__syncthreads()` inside a conditional branch (`if(threadIdx.x == 0) __syncthreads();`) and explain why this causes a deadlock.

Thinking:
- The CUDA programming model requires that all threads in a warp/block execute identical instructions, especially when synchronizing.
- `__syncthreads()` is a barrier that forces all threads in a block to wait until every thread has reached that point.
- If we guard a `__syncthreads()` with an `if(threadIdx.x == 0)`, only thread 0 will execute the barrier; all other threads will skip it.
- Threads that skip the barrier will continue executing beyond it, while thread 0 will be waiting for the others that never call the barrier.
- This mismatch leads to a deadlock: thread 0 waits forever, never progressing, and the rest of the threads also wait indefinitely because they are stuck in the same code path, not reaching the barrier.
- In practice, this causes the kernel to hang and may trigger a CUDA error like "invalid synchronization" or simply hang at runtime.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void deadlockKernel(int *d_out)
{
    // All threads write to output to keep it non-empty
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_out[idx] = idx;

    // Conditional barrier that causes deadlock
    if (threadIdx.x == 0)
    {
        __syncthreads(); // Only thread 0 reaches this, others skip
    }

    // Continue with some dummy work
    d_out[idx] = idx * 2;
}

int main(void)
{
    const int N = 256;
    int *h_out = (int *)malloc(N * sizeof(int));
    int *d_out;
    cudaMalloc((void **)&d_out, N * sizeof(int));

    // Launch kernel with one block of N threads
    deadlockKernel<<<1, N>>>(d_out);

    // Attempt to copy results back (this will hang due to deadlock)
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // If the kernel completed, print results
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_out);
    free(h_out);
    return 0;
}
