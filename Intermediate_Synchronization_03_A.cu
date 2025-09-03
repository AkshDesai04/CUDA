```cuda
/* 
The incorrect increment is a 'read-modify-write' race. 
Explain the sequence: all threads read the old value, all threads compute the new value, all threads write their new value. 
The last one to write wins. 

The following CUDA program demonstrates this race condition by launching a kernel where each thread attempts 
to increment a single shared counter using plain assignment (i.e. non‑atomic).  All threads first load the 
current value of the counter into a register, increment it locally, and then store the result back.  Because 
the loads happen before any thread has written the new value, all threads see the same old value.  The 
computation phase is identical for all threads, so each thread calculates the same new value.  When the 
store phase begins, whichever thread reaches the write instruction last overwrites all the previous 
writes, effectively erasing the increments performed by the other threads.  As a result, the final value of 
the counter is usually far less than the number of threads that executed the kernel.

To compare, the program also launches a second kernel that performs the same increment operation but 
uses the built‑in `atomicAdd` function.  This guarantees that each increment is performed atomically, 
ensuring the counter reflects the correct number of increments.  The host prints both results, showing 
the discrepancy caused by the race.  
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that performs a non-atomic increment of a global counter
__global__ void incrementNonAtomic(int *counter) {
    // Load the current value
    int temp = *counter;
    // Increment locally
    temp += 1;
    // Store back
    *counter = temp;
}

// Kernel that performs an atomic increment of a global counter
__global__ void incrementAtomic(int *counter) {
    atomicAdd(counter, 1);
}

int main() {
    const int numThreads = 1024;
    const int numBlocks = 1;
    const int totalThreads = numThreads * numBlocks;

    int *dev_counter;
    int host_counter;

    // Allocate device memory for the counter
    cudaMalloc((void**)&dev_counter, sizeof(int));

    // ------------------- Non-atomic increment test -------------------
    // Initialize counter to zero
    cudaMemset(dev_counter, 0, sizeof(int));

    // Launch kernel
    incrementNonAtomic<<<numBlocks, numThreads>>>(dev_counter);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&host_counter, dev_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Non-atomic increment result: %d (expected %d)\n", host_counter, totalThreads);

    // ------------------- Atomic increment test -------------------
    // Initialize counter to zero
    cudaMemset(dev_counter, 0, sizeof(int));

    // Launch kernel
    incrementAtomic<<<numBlocks, numThreads>>>(dev_counter);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&host_counter, dev_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Atomic increment result: %d (expected %d)\n", host_counter, totalThreads);

    // Clean up
    cudaFree(dev_counter);

    return 0;
}
```