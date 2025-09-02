```cpp
/*
Aim of the program (word to word):
Use `cudaMemGetInfo` to query the amount of free and total memory on the device. Print these values before and after the deliberate leak.

Thinking:
The goal is to demonstrate how the CUDA runtime reports device memory usage. We will
1. Query the device memory with cudaMemGetInfo() before any allocations.
2. Print the free and total memory values in bytes.
3. Allocate a sizable chunk of device memory without freeing it – this simulates a memory leak.
4. Query the memory again after the allocation and print the new values.
5. Optionally allocate several chunks to emphasize the effect.
6. Clean up by freeing the allocated memory (although the prompt says "deliberate leak", freeing afterward
   is harmless and keeps the program well-behaved).
7. Use a simple error-checking macro to report any CUDA API failures.
*/

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)

// Function to print free and total device memory
void printDeviceMemory(const char* msg) {
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    printf("%s\n", msg);
    printf("  Free memory:  %zu bytes\n", freeMem);
    printf("  Total memory: %zu bytes\n", totalMem);
}

int main() {
    // Print memory before any allocation
    printDeviceMemory("Before allocation:");

    // Deliberate memory leak: allocate a large chunk but do not free
    const size_t leakSize = 512 * 1024 * 1024; // 512 MB
    void* devPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devPtr, leakSize));

    // Optionally allocate more chunks to make the leak more obvious
    const size_t anotherLeakSize = 256 * 1024 * 1024; // 256 MB
    void* devPtr2 = nullptr;
    CUDA_CHECK(cudaMalloc(&devPtr2, anotherLeakSize));

    // Print memory after the deliberate leak
    printDeviceMemory("After deliberate leak:");

    // Clean up (not strictly necessary for a "leak", but good practice)
    CUDA_CHECK(cudaFree(devPtr));
    CUDA_CHECK(cudaFree(devPtr2));

    // Final memory check
    printDeviceMemory("After freeing leaked memory:");

    return 0;
}
```