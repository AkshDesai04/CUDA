```cpp
/* 
The lock acquisition looks like: `while(atomicCAS(lock_ptr, 0, 1) != 0);`.

This program demonstrates how a simple spinlock can be implemented in CUDA using the atomicCAS
instruction. The aim of the program is to show the use of the lock acquisition pattern mentioned
above in a real CUDA kernel.  We allocate a global lock variable (stored in device memory) and a
shared counter.  Each thread will repeatedly:

1. Acquire the lock by performing a busy‑wait loop that repeatedly executes
   `atomicCAS(&lock, 0, 1)`.  The loop continues until the CAS returns 0, indicating that
   the lock was free and the thread has now taken it.
2. Increment the shared counter inside the critical section.
3. Release the lock by writing 0 back to the lock variable with `atomicExch(&lock, 0)`.

After all threads finish, the host copies the counter back from device memory and prints
the final value.  Correctness is verified by comparing the final counter value with the
expected number of increments (threads × iterations).  The code also includes a simple
macro for error checking CUDA API calls.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple macro to check CUDA errors
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that increments a global counter protected by a spinlock
__global__ void increment_with_lock(int *counter, int *lock, int increments_per_thread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < increments_per_thread; ++i)
    {
        // Acquire lock
        while (atomicCAS(lock, 0, 1) != 0)
            ; // spin until the lock is acquired

        // Critical section: increment shared counter
        (*counter)++;

        // Release lock
        atomicExch(lock, 0);
    }
}

int main()
{
    const int num_blocks = 4;
    const int threads_per_block = 256;
    const int increments_per_thread = 1000;
    const int total_increments_expected = num_blocks * threads_per_block * increments_per_thread;

    // Allocate device memory for counter and lock
    int *d_counter = nullptr;
    int *d_lock = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lock, sizeof(int)));

    // Initialize counter and lock to 0
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_lock, 0, sizeof(int)));

    // Launch kernel
    increment_with_lock<<<num_blocks, threads_per_block>>>(d_counter, d_lock, increments_per_thread);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    int h_counter = 0;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Final counter value: %d (expected %d)\n", h_counter, total_increments_expected);

    // Clean up
    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaFree(d_lock));

    return 0;
}
```