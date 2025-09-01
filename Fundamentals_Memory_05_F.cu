```cpp
/* 
Aim: Write a program that allocates 10 small chunks of memory and stores their pointers in an array. Then, loop through the array to free them all.

Thinking:
- The task is straightforward: allocate 10 device memory blocks, keep their pointers, then free them.
- Use CUDA runtime API: cudaMalloc for allocation, cudaFree for deallocation.
- Choose a small size, e.g., 256 bytes per chunk, to keep the program lightweight.
- Store pointers in a host array of type `void*` or `char*`.
- Wrap CUDA calls in a simple error-checking macro to make debugging easier.
- Use a loop to free each pointer.
- Add simple console output to confirm allocation and deallocation steps.
- Ensure the code is a valid .cu file, compiles with nvcc, and runs on any system with CUDA installed.
- The program will be self-contained, with includes, main function, and error checking.

*/

#include <iostream>
#include <cuda_runtime.h>

// Simple macro to check CUDA errors
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA error in " << __FILE__            \
                      << " at line " << __LINE__ << ": "        \
                      << cudaGetErrorString(err) << std::endl;   \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main() {
    const int NUM_CHUNKS = 10;
    const size_t CHUNK_SIZE = 256; // bytes per chunk

    // Allocate an array to hold device pointers
    void* d_ptrs[NUM_CHUNKS];

    // Allocate device memory for each chunk
    std::cout << "Allocating " << NUM_CHUNKS << " device memory chunks..." << std::endl;
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        CHECK_CUDA(cudaMalloc(&d_ptrs[i], CHUNK_SIZE));
        std::cout << "  Chunk " << i << " allocated at device pointer " << d_ptrs[i] << std::endl;
    }

    // Perform any operations on the device memory here (optional)

    // Free the allocated memory
    std::cout << "Freeing allocated device memory..." << std::endl;
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        CHECK_CUDA(cudaFree(d_ptrs[i]));
        std::cout << "  Chunk " << i << " freed." << std::endl;
    }

    std::cout << "All memory freed successfully." << std::endl;
    return 0;
}
```