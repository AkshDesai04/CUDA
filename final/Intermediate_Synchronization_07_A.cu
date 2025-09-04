/*
The lock acquisition looks like: `while(atomicCAS(lock_ptr, 0, 1) != 0);`

Thinking:
- The goal is to illustrate how a simple spinlock can be implemented on the GPU using
  `atomicCAS`.  The idiom `while(atomicCAS(lock_ptr, 0, 1) != 0);` repeatedly attempts
  to change the lock value from 0 (unlocked) to 1 (locked) until it succeeds.
- In this program we allocate a single integer lock in device global memory
  (`d_lock`).  Threads in a kernel try to acquire the lock before entering a
  critical section where they increment a shared counter (`d_counter`).  After
  the increment the lock is released by setting it back to 0 with `atomicExch`
  (which is effectively the same as `atomicCAS(&lock, 1, 0)` but clearer).
- We also pass `increments_per_thread` so that each thread performs several
  lock acquisitions, making contention more visible.  The final counter value
  should equal `num_threads * increments_per_thread`.
- `atomicCAS` guarantees atomicity on GPU, but it is a busy-wait loop
  (spinlock).  In real applications this can waste GPU cycles; it is used here
  only for educational purposes.
- The code uses `cudaMemset` to initialise the lock to 0 and the counter to 0.
  After kernel launch, we copy back the counter to host and print it.
- We also include basic error checking for CUDA calls to make debugging easier.
- No other synchronization primitives (e.g., `__syncthreads`) are required
  because the lock protects the critical section across thread blocks.
- The program is self‑contained and compiles as a single .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that demonstrates lock acquisition using atomicCAS.
__global__ void lock_demo(int *lock, int *counter, int increments_per_thread)
{
    // Each thread performs several increments protected by the lock.
    for (int i = 0; i < increments_per_thread; ++i)
    {
        // Acquire the lock: keep trying until we succeed.
        // atomicCAS returns the old value, so we loop until the old value is 0.
        while (atomicCAS(lock, 0, 1) != 0)
        {
            // Busy-wait; in real code you might want to yield or add backoff.
        }

        // Critical section: increment the shared counter.
        atomicAdd(counter, 1);

        // Release the lock by setting it back to 0.
        atomicExch(lock, 0);
    }
}

int main()
{
    const int num_blocks = 4;
    const int threads_per_block = 128;
    const int num_threads = num_blocks * threads_per_block;
    const int increments_per_thread = 100;

    int *d_lock, *d_counter;
    int h_counter = 0;

    // Allocate device memory for lock and counter.
    cudaError_t err = cudaMalloc((void **)&d_lock, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for lock failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void **)&d_counter, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for counter failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_lock);
        return 1;
    }

    // Initialise lock and counter to 0.
    err = cudaMemset(d_lock, 0, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset for lock failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_lock); cudaFree(d_counter);
        return 1;
    }

    err = cudaMemset(d_counter, 0, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset for counter failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_lock); cudaFree(d_counter);
        return 1;
    }

    // Launch the kernel.
    lock_demo<<<num_blocks, threads_per_block>>>(d_lock, d_counter, increments_per_thread);

    // Wait for GPU to finish.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_lock); cudaFree(d_counter);
        return 1;
    }

    // Copy result back to host.
    err = cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_lock); cudaFree(d_counter);
        return 1;
    }

    // Expected result is num_threads * increments_per_thread.
    int expected = num_threads * increments_per_thread;
    printf("Result counter = %d, expected = %d\n", h_counter, expected);

    // Clean up.
    cudaFree(d_lock);
    cudaFree(d_counter);

    return 0;
}
