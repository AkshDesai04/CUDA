```cuda
/* 
Aim of the program:
Write a program that allocates 10 small chunks of memory and stores their pointers in an array. Then, loop through the array to free them all.

Thinking process:
1. The task requires a CUDA program written in C. 
2. The phrase "allocates 10 small chunks of memory" can be interpreted in a CUDA context as device memory allocations using cudaMalloc. 
3. We will create an array of 10 device pointers on the host, allocate a small chunk (e.g., 256 bytes) for each pointer, and store the device pointers in the array. 
4. After allocation, we loop through the array and free each chunk with cudaFree. 
5. To make the program robust, we’ll include basic error checking for each CUDA API call. 
6. We will also print status messages to the console so that when the program is run, it is clear that allocation and deallocation have succeeded. 
7. The program will be a simple standalone executable that can be compiled with nvcc. 
8. No additional libraries or GPU kernels are required; the focus is purely on memory management. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void) {
    const int NUM_CHUNKS = 10;
    const size_t CHUNK_SIZE = 256; // 256 bytes per chunk

    // Allocate an array on the host to store device pointers
    void* dev_ptrs[NUM_CHUNKS];

    // Allocate each chunk on the device
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        CUDA_CHECK(cudaMalloc(&dev_ptrs[i], CHUNK_SIZE));
        printf("Allocated chunk %d at device address %p\n", i, dev_ptrs[i]);
    }

    // Free all allocated chunks
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        CUDA_CHECK(cudaFree(dev_ptrs[i]));
        printf("Freed chunk %d at device address %p\n", i, dev_ptrs[i]);
    }

    printf("All chunks have been freed successfully.\n");
    return 0;
}
```