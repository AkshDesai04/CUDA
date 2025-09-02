/*
Aim: Implement a simple spin-lock in global memory using atomicCAS (Compare-And-Swap). A thread tries to acquire the lock by swapping a 0 to a 1. If it succeeds, it enters a critical section. It releases the lock by writing 0 back. (Note: This is often inefficient on GPUs but is a good exercise).

Thinking:
- A global integer variable will serve as the lock; 0 indicates unlocked, 1 indicates locked.
- Two device helper functions are defined: `spinLockLock` performs a busy‑wait loop using `atomicCAS` until it successfully swaps 0 → 1; `spinLockUnlock` releases the lock by writing 0 back (using `atomicExch` for atomicity).
- A simple global counter is also defined; each thread will increment this counter while holding the lock to demonstrate mutual exclusion.
- Inside the kernel each thread calls `spinLockLock`, enters the critical section, updates the counter, prints its thread ID and the current counter value using `printf`, then calls `spinLockUnlock`.
- On the host side we allocate the lock and counter on the device, initialize them to 0, launch the kernel with a reasonable number of threads, copy back the counter value, and print the final count.
- The program uses standard CUDA APIs and can be compiled with `nvcc -o spinlock spinlock.cu`. It also performs minimal error checking for clarity.

The code below implements the described approach.*/

#include <cstdio>
#include <cuda_runtime.h>

#define NUM_THREADS 256
#define NUM_BLOCKS 1

/* Global lock variable in device memory */
__device__ int g_lock = 0;
/* Global counter variable in device memory */
__device__ int g_counter = 0;

/* Spin‑lock acquire function using atomicCAS */
__device__ void spinLockLock(int *lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // Busy wait until the lock is free
    }
}

/* Spin‑lock release function */
__device__ void spinLockUnlock(int *lock) {
    atomicExch(lock, 0);
}

/* Kernel where each thread attempts to acquire the lock,
   updates a global counter, and then releases the lock */
__global__ void testKernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Acquire the lock */
    spinLockLock(&g_lock);

    /* Critical section */
    g_counter += 1;
    printf("Thread %d acquired lock, counter=%d\n", tid, g_counter);

    /* Release the lock */
    spinLockUnlock(&g_lock);
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main() {
    /* Launch the kernel */
    testKernel<<<NUM_BLOCKS, NUM_THREADS>>>();
    CUDA_CHECK(cudaGetLastError());

    /* Wait for kernel to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy final counter value back to host */
    int host_counter = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&host_counter, g_counter, sizeof(int), 0, cudaMemcpyDeviceToHost));

    printf("Final counter value: %d (expected %d)\n", host_counter, NUM_THREADS);

    /* Reset the device */
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
