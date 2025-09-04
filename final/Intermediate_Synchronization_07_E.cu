```cuda
/*
Explain why spin-locks are generally a bad idea on GPUs. (Massive thread counts cause extreme contention; threads that are spinning are wasting execution resources and preventing other warps from running).

The aim of this program is to demonstrate, in a minimal CUDA kernel, how a spin-lock works and why it can be problematic on GPUs.  In GPU programming, thousands of threads execute in warps.  A spin-lock forces threads to repeatedly check a flag and wait until it becomes free.  Because warps are scheduled cooperatively, if one warp is busy spinning it can block the scheduler from dispatching other warps, leading to under‑utilization of the GPU.  Moreover, with the massive parallelism of GPUs, many threads may simultaneously attempt to acquire the lock, causing a burst of atomic operations and cache traffic that further degrades performance.  The code below shows a simple global spin-lock implemented with atomicCAS/atomicExch, and a kernel that many threads try to use.  Running it will illustrate the contention and the wasteful spinning.

Thinking about the implementation:
- We need a flag in global memory that all threads can access.  It will be initialized to 0 (unlocked).
- The spin_lock function uses atomicCAS to set the flag from 0 to 1.  If the CAS fails, the thread spins.
- The spin_unlock simply sets the flag back to 0 using atomicExch.
- The kernel will be launched with many threads (e.g., 1024 per block, 10 blocks) to increase contention.
- Inside the critical section we perform an atomicAdd to a shared counter to demonstrate work done while holding the lock.
- After the critical section, we release the lock and print a message.  Using printf from device code requires nvcc with compute capability 2.0+.
- On the host side, we allocate the lock flag and counter, launch the kernel, copy the counter back, and free memory.
- This example does not optimize for performance; it is purely illustrative of spin-lock behavior on a GPU.

Compile with: nvcc -arch=sm_60 spin_lock_demo.cu -o spin_lock_demo
Run with: ./spin_lock_demo
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple spin lock implementation using atomic operations
__device__ void spin_lock(int *flag) {
    while (atomicCAS(flag, 0, 1) != 0) {
        // Busy wait (spin)
        // Optional: use __nanosleep to reduce GPU usage, but here we keep it tight for demonstration
    }
}

__device__ void spin_unlock(int *flag) {
    atomicExch(flag, 0);
}

// Kernel where each thread attempts to increment a global counter under a spin lock
__global__ void spinLockKernel(int *counter, int *lock) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Acquire the lock
    spin_lock(lock);

    // Critical section: increment the shared counter
    int old_val = atomicAdd(counter, 1);

    // Release the lock
    spin_unlock(lock);

    // Print the result (requires device printf)
    printf("Thread %d incremented counter from %d to %d\n",
           thread_id, old_val, old_val + 1);
}

int main() {
    const int NUM_BLOCKS = 10;
    const int THREADS_PER_BLOCK = 1024;
    const int NUM_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;

    // Allocate memory on device for counter and lock flag
    int *d_counter;
    int *d_lock;
    cudaMalloc((void**)&d_counter, sizeof(int));
    cudaMalloc((void**)&d_lock, sizeof(int));

    // Initialize counter and lock to 0
    cudaMemset(d_counter, 0, sizeof(int));
    cudaMemset(d_lock, 0, sizeof(int));

    // Launch the kernel
    spinLockKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_counter, d_lock);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy back the final counter value
    int h_counter = 0;
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nFinal counter value: %d (expected %d)\n", h_counter, NUM_THREADS);

    // Clean up
    cudaFree(d_counter);
    cudaFree(d_lock);

    return 0;
}
```