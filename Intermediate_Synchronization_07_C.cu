```cuda
/*
Use this lock to protect the non-atomic increment from the earlier race condition exercise.
Only one thread at a time will be able to perform the read-modify-write.

Thinking:
- In CUDA, many threads can concurrently access and modify shared data leading to race conditions.
- An earlier exercise probably involved each thread incrementing a global counter without any synchronization,
  which results in lost updates because the increment operation is not atomic.
- To fix this, we implement a simple spinlock using atomicCAS (compare-and-swap). 
  The lock is represented by an integer variable (`lock`) where 0 = unlocked, 1 = locked.
- Each thread acquires the lock by repeatedly calling atomicCAS(&lock, 0, 1) until it succeeds.
  Once inside the critical section, the thread performs the non-atomic increment (`counter++`).
  After the increment, the thread releases the lock by setting `lock` back to 0 using atomicExch.
- This guarantees that only one thread can perform the read-modify-write at a time.
- The program demonstrates both the incorrect result (without a lock) and the correct result (with the lock).
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCKS  8
#define THREADS_PER_BLOCK  256
#define NUM_INCS  1000   // number of increments each thread performs

__device__ int counter = 0;
__device__ int lock    = 0;

// Kernel that increments counter without any synchronization (races!)
__global__ void inc_kernel_no_lock()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < NUM_INCS; ++i)
    {
        counter++;  // non-atomic, can be lost
    }
}

// Kernel that increments counter protected by a simple spinlock
__global__ void inc_kernel_lock()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < NUM_INCS; ++i)
    {
        // acquire lock
        while (atomicCAS(&lock, 0, 1) != 0)
            ;  // busy-wait until lock becomes 0

        // critical section
        counter++;

        // release lock
        atomicExch(&lock, 0);
    }
}

// Utility function to reset device globals
void reset_device_globals()
{
    int zero = 0;
    cudaMemcpyToSymbol(counter, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(lock,    &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
}

int main()
{
    int host_counter = 0;
    size_t total_increments = (size_t)BLOCKS * THREADS_PER_BLOCK * NUM_INCS;

    // Run kernel without lock to show race condition
    reset_device_globals();
    inc_kernel_no_lock<<<BLOCKS, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&host_counter, counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    printf("Counter without lock (expected %zu): %d\n", total_increments, host_counter);

    // Run kernel with lock to ensure correct increment
    reset_device_globals();
    inc_kernel_lock<<<BLOCKS, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&host_counter, counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    printf("Counter with lock (expected %zu): %d\n", total_increments, host_counter);

    cudaDeviceReset();
    return 0;
}
```