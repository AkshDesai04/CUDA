/*
 * Aim: Write a program that deliberately leaks memory on the device by allocating it but never calling cudaFree.
 * 
 * Thinking:
 * - The program should use CUDA runtime API to allocate device memory with cudaMalloc.
 * - To observe the leak, we will allocate several chunks of memory in a loop and store the pointers in a std::vector so we never free them.
 * - After each allocation we print the pointer and the size so the user can see the growth of usage.
 * - We sleep briefly between allocations to give the system time to update the GPU memory stats.
 * - Finally we pause the program (waiting for user input) so the user can run `nvidia-smi` in another terminal to monitor the memory usage while the program is still running.
 * - We do not call cudaFree on any of the allocations, so the memory remains allocated until the process exits.
 * - When the process exits, the CUDA driver will clean up automatically, but the memory leak is still demonstrated during the run.
 * 
 * Note:
 * - Each run of the program will allocate memory but never free it within that run.
 * - Because CUDA frees device memory at process termination, running the program multiple times will not accumulate memory across runs; instead, each run will show increasing usage while it is running.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>

int main()
{
    const size_t allocSize = 100 * 1024 * 1024; // 100 MB per allocation
    const int numAllocations = 10;             // total allocations

    std::vector<void*> devPtrs;
    devPtrs.reserve(numAllocations);

    printf("Starting memory leak demonstration...\n");
    printf("Allocating %zu MB %d times without freeing.\n", allocSize / (1024 * 1024), numAllocations);

    for (int i = 0; i < numAllocations; ++i)
    {
        void* d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, allocSize);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc failed at allocation %d: %s\n", i, cudaGetErrorString(err));
            break;
        }
        devPtrs.push_back(d_ptr);
        printf("Allocation %2d: %zu bytes at %p\n", i + 1, allocSize, d_ptr);

        // Give the system a moment to update memory stats
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    printf("\nAll allocations complete.\n");
    printf("Memory has been allocated but not freed.\n");
    printf("You can now run 'nvidia-smi' in another terminal to observe GPU memory usage.\n");
    printf("Press Enter to exit (memory will be freed by the driver upon termination)...\n");
    getchar();

    // No cudaFree calls: intentional memory leak
    return 0;
}
