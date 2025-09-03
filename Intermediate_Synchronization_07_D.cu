/*
Aim: Profile the spin-lock version against the `atomicAdd` version. The spin-lock will be much, much slower due to serialization.

Thinking:
To compare the performance of a simple spin‑lock against CUDA's built‑in `atomicAdd`, we create two kernels that both perform the same logical work: each thread increments a global counter a fixed number of times.  
1. **Spin‑lock kernel**  
   - Uses a global integer `lock` initialized to 0.  
   - Each increment is protected by a busy‑wait loop that calls `atomicCAS(lock, 0, 1)` until it succeeds.  
   - After the protected increment, the thread releases the lock by setting `lock` back to 0.  
   - Because all threads contend for the same lock, operations are serialized, producing a slow execution time.  

2. **AtomicAdd kernel**  
   - Each increment is performed by calling `atomicAdd(&counter, 1)` directly.  
   - CUDA schedules atomic operations in a way that allows some level of concurrency and usually performs better than a hand‑written spin‑lock.  

To make the serialization noticeable we use a moderate number of threads (e.g., 8192) and have each thread perform a large number of increments (e.g., 1000).  
The program measures the elapsed GPU time for each kernel using CUDA events, copies the final counter value back to host memory, and prints the results.  

The expected outcome is that the spin‑lock version will take significantly longer than the atomicAdd version, illustrating the cost of serialization when using a hand‑rolled lock in CUDA.  
*/

#include <stdio.h>
#include <cuda.h>

// Spin‑lock kernel: each thread increments the counter a fixed number of times using a global lock
__global__ void spinLockKernel(int *counter, int *lock, int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iterations; ++i)
    {
        // Acquire lock
        while (atomicCAS(lock, 0, 1) != 0)
        {
            // busy‑wait
        }

        // Critical section
        *counter = *counter + 1;  // simple increment; could also use atomicAdd but lock guarantees exclusivity

        // Release lock
        *lock = 0;
    }
}

// AtomicAdd kernel: each thread performs atomicAdd on the counter
__global__ void atomicAddKernel(int *counter, int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iterations; ++i)
    {
        atomicAdd(counter, 1);
    }
}

int main()
{
    const int threadsPerBlock = 256;
    const int numThreads = 8192; // total threads
    const int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    const int iterations = 1000; // increments per thread

    // Device memory
    int *d_counter = nullptr;
    int *d_lock = nullptr;
    cudaMalloc((void **)&d_counter, sizeof(int));
    cudaMalloc((void **)&d_lock, sizeof(int));

    // Initialize counter and lock to 0
    cudaMemset(d_counter, 0, sizeof(int));
    cudaMemset(d_lock, 0, sizeof(int));

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------- Spin‑lock kernel ----------
    cudaMemset(d_counter, 0, sizeof(int));
    cudaMemset(d_lock, 0, sizeof(int));

    cudaEventRecord(start, 0);
    spinLockKernel<<<blocks, threadsPerBlock>>>(d_counter, d_lock, iterations);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float spinTime = 0.0f;
    cudaEventElapsedTime(&spinTime, start, stop);

    int h_counter = 0;
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Spin‑lock kernel:\n");
    printf("  Time elapsed: %.3f ms\n", spinTime);
    printf("  Final counter value: %d (expected %d)\n", h_counter, numThreads * iterations);
    printf("\n");

    // ---------- AtomicAdd kernel ----------
    cudaMemset(d_counter, 0, sizeof(int));

    cudaEventRecord(start, 0);
    atomicAddKernel<<<blocks, threadsPerBlock>>>(d_counter, iterations);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float atomicTime = 0.0f;
    cudaEventElapsedTime(&atomicTime, start, stop);

    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("AtomicAdd kernel:\n");
    printf("  Time elapsed: %.3f ms\n", atomicTime);
    printf("  Final counter value: %d (expected %d)\n", h_counter, numThreads * iterations);
    printf("\n");

    // Clean up
    cudaFree(d_counter);
    cudaFree(d_lock);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
