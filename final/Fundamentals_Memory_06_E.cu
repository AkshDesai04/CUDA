```cpp
/*
Aim of the program:
Use `cudaMemGetInfo` to query the amount of free and total memory on the device. Print these values before and after the deliberate leak.

Thinking process:
1. Include necessary headers: <iostream> for console output and <cuda_runtime.h> for CUDA runtime API.
2. Define a helper macro `CHECK_CUDA` to simplify error handling. This macro will execute a CUDA runtime function, check its return value, and if an error occurs, print the error message and exit.
3. In `main()`, first query the device memory information using `cudaMemGetInfo(&free, &total)` and print the values. This gives us the baseline free and total memory before any allocations.
4. Allocate a chunk of device memory deliberately without freeing it. We'll use `cudaMalloc(&ptr, size)`. The size will be chosen to consume a noticeable amount of memory (e.g., 100 MB) but not too large to crash typical devices.
5. Query the memory again using `cudaMemGetInfo` after the allocation and print the values. The free memory should have decreased by approximately the allocation size (minus any overhead).
6. Since the assignment asks for a "deliberate leak", we intentionally omit the `cudaFree(ptr)` call. This demonstrates how leaked allocations affect available memory.
7. End the program normally. The operating system will reclaim the leaked device memory when the process exits, but the demonstration shows the effect during the program's execution.

The code is self‑contained and can be compiled with `nvcc`:

    nvcc -o memleak_demo memleak_demo.cu

Running the resulting executable will display the free/total memory before and after the deliberate allocation leak.
*/

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Function to print memory info
void printMemoryInfo(const std::string& prefix) {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&freeBytes, &totalBytes));
    std::cout << prefix << std::endl;
    std::cout << "  Free Memory : " << freeBytes << " bytes" << std::endl;
    std::cout << "  Total Memory: " << totalBytes << " bytes" << std::endl;
}

int main() {
    std::cout << "CUDA Device Memory Leak Demo" << std::endl;

    // 1. Print memory info before allocation
    printMemoryInfo("Before allocation:");

    // 2. Deliberate allocation that will not be freed
    const size_t allocationSize = 100 * 1024 * 1024; // 100 MB
    void* devPtr = nullptr;
    std::cout << "\nAllocating " << allocationSize << " bytes on the device (deliberate leak)." << std::endl;
    CHECK_CUDA(cudaMalloc(&devPtr, allocationSize));

    // 3. Print memory info after allocation
    std::cout << std::endl;
    printMemoryInfo("After allocation (before freeing):");

    // 4. Intentionally skip cudaFree to create a leak
    std::cout << "\nDeliberate leak: not freeing the allocated memory." << std::endl;

    // Program ends; the leaked memory will be reclaimed by the OS.
    return 0;
}
```