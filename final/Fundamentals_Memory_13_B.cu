```cu
/*
Aim of the program (exact wording from the prompt):
Write a loop that keeps allocating 1GB chunks of memory until `cudaMalloc` fails. Print how many GB were successfully allocated before failure.

Thinking:
1. Each allocation chunk is 1 GB = 1024 * 1024 * 1024 bytes.  This fits in a 64‑bit size_t.
2. We need to keep allocating until cudaMalloc returns an error.  We’ll use the return code directly to determine success.
3. Store each pointer in a vector so that we can free all allocations afterwards, preventing a memory leak on the device.
4. Count the number of successful allocations; each one represents 1 GB, so the count is the total GB allocated.
5. After the loop ends, print the count using printf.  If the first allocation fails, the count will be 0.
6. Finally, iterate over the vector and cudaFree each pointer to release the device memory.
7. The program will include <cuda_runtime.h> for CUDA APIs and <vector> for std::vector.  We’ll use <cstdio> for printf.
8. The program is self‑contained and compiles as a .cu file with nvcc.
*/

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

int main() {
    const size_t chunkSize = 1024ull * 1024ull * 1024ull; // 1 GB
    std::vector<void*> allocations;
    int count = 0;

    while (true) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, chunkSize);
        if (err != cudaSuccess) {
            // Allocation failed; break out of the loop.
            break;
        }
        allocations.push_back(ptr);
        count++;
    }

    printf("Successfully allocated %d GB before failure.\n", count);

    // Clean up all allocations
    for (void* ptr : allocations) {
        cudaFree(ptr);
    }

    return 0;
}
```