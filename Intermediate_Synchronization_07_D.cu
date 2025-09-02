```cpp
/*
Aim of the program:
Profile the spin-lock version against the `atomicAdd` version. The spin-lock will be much, much slower due to serialization.

Thinking:
The task is to create a CUDA C program that compares two ways of synchronizing access to a shared counter among many threads:
1. Using a spin-lock implemented with atomicCAS and a while loop.
2. Using CUDA's built-in atomicAdd.

To make the comparison meaningful, each kernel will increment a single integer counter many times (e.g., 10,000 times per thread) in a loop. The spin-lock kernel will serialize access to the counter by acquiring a lock before each increment, whereas the atomicAdd kernel will perform the increment in a single atomic operation. We will time both kernels with CUDA events to measure execution time, print the times, and verify that the final counter value is correct after each kernel.

Key considerations:
- The lock is a single integer located in global memory, initialized to 0 (unlocked).
- The spin-lock acquisition uses `atomicCAS(lock, 0, 1)`; if it returns non-zero, the thread spins until the lock becomes available.
- The critical section in the spin-lock kernel is simply `(*counter)++;` followed by releasing the lock via `atomicExch(lock, 0);`.
- The atomic kernel simply does `atomicAdd(counter, 1);` in its loop.
- We will use `cudaEvent_t` for precise timing.
- After each kernel, we copy the counter back to host and reset it to zero before the next run.
- Error checking is performed after each CUDA call.
- The program is self-contained and can be compiled with `nvcc`. 
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Number of threads per block
#define THREADS_PER_BLOCK 256
// Number of blocks
#define NUM_BLOCKS 64
// Number of increments each thread performs
#define INCREMENTS_PER_THREAD 10000

// Spin-lock kernel: each thread increments the counter INCREMENTS_PER_THREAD times
__global__ void spin_lock_kernel(int *counter, int *lock)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < INCREMENTS_PER_THREAD; ++i)
    {
        // Acquire lock
        while (atomicCAS(lock, 0, 1) != 0) {
            // busy-wait
        }

        // Critical section
        (*counter)++;

        // Release lock
        atomicExch(lock, 0);
    }
}

// AtomicAdd kernel: each thread increments the counter INCREMENTS_PER_THREAD times
__global__ void atomic_kernel(int *counter)
{
    for (int i = 0; i < INCREMENTS_PER_THREAD; ++i)
    {
        atomicAdd(counter, 1);
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",   \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),       \
                    cudaGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    // Device pointers
    int *d_counter = nullptr;
    int *d_lock = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_lock, sizeof(int)));

    // Host variable for final counter
    int h_counter = 0;

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Spin-lock version
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lock, 0, sizeof(int))); // lock unlocked

    CUDA_CHECK(cudaEventRecord(start, 0));
    spin_lock_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_counter, d_lock);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float spin_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&spin_time_ms, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Spin-lock result: counter = %d\n", h_counter);

    // Reset counter
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lock, 0, sizeof(int)));

    // AtomicAdd version
    CUDA_CHECK(cudaEventRecord(start, 0));
    atomic_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_counter);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float atomic_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&atomic_time_ms, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("AtomicAdd result: counter = %d\n", h_counter);

    // Print timings
    printf("Spin-lock time: %.3f ms\n", spin_time_ms);
    printf("AtomicAdd time: %.3f ms\n", atomic_time_ms);

    // Clean up
    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaFree(d_lock));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```