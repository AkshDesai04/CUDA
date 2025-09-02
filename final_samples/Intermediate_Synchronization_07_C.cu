/*
Aim: Use this lock to protect the non-atomic increment from the earlier race condition exercise. Only one thread at a time will be able to perform the read-modify-write.

[Thinking]
To serialize access to a shared counter that is incremented without atomic operations, we need a mutual exclusion primitive that can be used on the GPU. A common choice is a spinlock implemented with the atomicCAS (compare-and-swap) instruction. The lock is represented as an integer where 0 means unlocked and 1 means locked. Threads repeatedly try to set the lock from 0 to 1; if they fail, they spin until the lock becomes available. Once a thread has the lock, it can safely read, modify, and write back the counter. After the critical section, the thread releases the lock by setting it back to 0 using atomicExch. This ensures that only one thread at a time performs the read-modify-write, eliminating the race condition observed previously.

The program defines a global counter and lock, a kernel that increments the counter using the lock, and host code that launches the kernel with many threads. After execution, the final counter value is printed; it should match the total number of threads launched, confirming correct serialization.
*/

#include <iostream>
#include <cuda_runtime.h>

// Global counter and lock on the device
__device__ int d_counter = 0;
__device__ int d_lock    = 0;

// Kernel that increments the counter using a spinlock
__global__ void incrementWithLock()
{
    // Acquire lock: try to change from 0 to 1
    while (atomicCAS(&d_lock, 0, 1) != 0) {
        // Busy wait
    }

    // Critical section: non-atomic read-modify-write
    int temp = d_counter;
    temp = temp + 1;
    d_counter = temp;

    // Release lock
    atomicExch(&d_lock, 0);
}

int main()
{
    // Number of threads and blocks
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = 256;
    const int totalThreads = threadsPerBlock * blocksPerGrid;

    // Launch kernel
    incrementWithLock<<<blocksPerGrid, threadsPerBlock>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sync error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy final counter value back to host
    int h_counter = 0;
    err = cudaMemcpyFromSymbol(&h_counter, d_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Memcpy error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Final counter value: " << h_counter << std::endl;
    std::cout << "Expected value:    " << totalThreads << std::endl;

    return 0;
}
