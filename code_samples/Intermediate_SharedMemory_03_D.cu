/*
Place a `__syncthreads()` inside a conditional branch (`if(threadIdx.x == 0) __syncthreads();`)
and explain why this causes a deadlock.

Thinking:
In CUDA, `__syncthreads()` is a barrier that synchronizes all threads within a block.
Every thread in the block must execute the barrier; otherwise the other threads will
wait indefinitely for the missing threads, causing a deadlock.  Placing the barrier
inside an `if(threadIdx.x == 0)` causes only thread 0 to execute the barrier,
while the remaining threads skip it.  Thread 0 will block at the barrier waiting
for the other threads, which have already exited the kernel.  Consequently, the
kernel never terminates and the host program hangs on `cudaDeviceSynchronize()`.
This example demonstrates the problem and prints messages from each thread to
illustrate the execution flow before the deadlock occurs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void deadlockKernel()
{
    int tid = threadIdx.x;

    // All threads print a message before the conditional barrier
    printf("Thread %d: starting execution.\n", tid);

    // Conditional barrier: only thread 0 reaches __syncthreads()
    if (tid == 0) {
        printf("Thread %d: before __syncthreads().\n", tid);
        __syncthreads();               // <--- Only thread 0 calls this
        printf("Thread %d: after __syncthreads().\n", tid);
    }

    // Rest of the code (executed by all threads) - here we just print
    printf("Thread %d: finishing execution.\n", tid);
}

int main()
{
    const int threadsPerBlock = 32;
    const int blocksPerGrid = 1;

    // Launch the kernel
    deadlockKernel<<<blocksPerGrid, threadsPerBlock>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Attempt to synchronize with the device; this will block because the kernel
    // never terminates due to the deadlock introduced by the conditional barrier.
    printf("Host: waiting for kernel to finish (this will hang)...\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Host: kernel finished successfully.\n");
    }

    return 0;
}
