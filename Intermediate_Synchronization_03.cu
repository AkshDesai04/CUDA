```cpp
/*
 * Aim of the program:
 * Explain what a 'race condition' is in the context of CUDA. 
 * Write a kernel where multiple threads attempt to increment the same global memory location without atomics (`output[0] = output[0] + 1`). 
 * Run it and show that the final result is incorrect.
 *
 * Thinking:
 * 1. In CUDA, multiple threads can execute concurrently on a GPU. 
 *    When several threads try to update the same memory location without proper synchronization (e.g., atomic operations), the updates can interleave in an unpredictable order. 
 *    Each thread typically performs the operation in three steps: read the value, modify it (add 1), and write it back. 
 *    If two or more threads perform these steps at the same time, a thread may overwrite a value written by another thread before that other thread has had a chance to write its own incremented value. 
 *    This leads to lost updates and the final value being less than the number of increments performed. 
 *    This phenomenon is known as a "race condition" because the outcome depends on the relative timing of the threads, which is nondeterministic.
 *
 * 2. To demonstrate, we allocate a single-element array `output` in global memory and launch a large number of threads. 
 *    Each thread executes `output[0] = output[0] + 1;` without using `atomicAdd`. 
 *    Because the memory location is shared, all threads race to read and write the same value.
 *
 * 3. We expect the final value to equal the total number of threads launched. 
 *    Due to the race condition, the actual value will be noticeably smaller. 
 *    Printing both expected and actual values will illustrate the bug.
 *
 * 4. The code below includes basic error checking and prints the discrepancy. 
 *    Compile with `nvcc race_condition_demo.cu -o race_condition_demo` and run.
 */

#include <iostream>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << cudaGetErrorString(err) << std::endl;     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel where all threads increment the same global memory location without atomics
__global__ void incrementNoAtomic(int *output)
{
    // All threads perform the same read-modify-write on output[0]
    // No synchronization or atomic operations are used.
    output[0] = output[0] + 1;
}

int main()
{
    const int numThreads = 1024 * 1024;  // One million increments
    const int blockSize = 256;
    const int gridSize = (numThreads + blockSize - 1) / blockSize;

    // Allocate memory for output on device
    int *d_output;
    CHECK_CUDA(cudaMalloc((void **)&d_output, sizeof(int)));

    // Initialize output[0] to zero
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int)));

    // Launch kernel
    incrementNoAtomic<<<gridSize, blockSize>>>(d_output);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    int h_output;
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_output));

    // Expected value is numThreads
    std::cout << "Expected final value: " << numThreads << std::endl;
    std::cout << "Actual final value:   " << h_output << std::endl;
    std::cout << "Difference:           " << (numThreads - h_output) << std::endl;

    return 0;
}
```