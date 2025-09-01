```cuda
/*
Aim of the program:
Write a loop that keeps allocating 1GB chunks of memory until `cudaMalloc` fails. Print how many GB were successfully allocated before failure.

Thought Process:
- We need to repeatedly call cudaMalloc with a chunk size of 1GB (1024 * 1024 * 1024 bytes).
- The allocations should be stored in a container (e.g., std::vector<void*>). This ensures we keep the memory allocated until the program ends.
- On each iteration, check the return value of cudaMalloc. If it is not cudaSuccess, we break out of the loop.
- Keep a counter of how many successful allocations we performed.
- After the loop, print the total number of GB allocated (counter * 1GB).
- Finally, free all allocated memory to avoid leaks and call cudaDeviceReset() to clean up the device.
- Use standard CUDA runtime API functions and standard C++ headers for ease of implementation.
- Ensure the code compiles as a .cu file and can be built with nvcc.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

int main() {
    const size_t chunkSize = 1ULL << 30; // 1GB in bytes
    std::vector<void*> allocations;
    size_t count = 0;

    while (true) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, chunkSize);
        if (err != cudaSuccess) {
            // Allocation failed, exit loop
            break;
        }
        allocations.push_back(ptr);
        ++count;
    }

    printf("%zu GB successfully allocated before failure.\n", count);

    // Free allocated memory
    for (void* ptr : allocations) {
        cudaFree(ptr);
    }

    // Reset device
    cudaDeviceReset();

    return 0;
}
```