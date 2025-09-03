/*
Write a program that allocates 10 small chunks of memory and stores their pointers in an array. Then, loop through the array to free them all.

My thinking process for implementing this program in CUDA C is as follows:

1. Decide which type of memory to allocate.  
   Since the problem is framed in a CUDA context, the natural choice is to use device memory allocated with `cudaMalloc`.  
   The pointers will be stored in a host array of `void*`, because `cudaMalloc` returns device pointers that can be passed around by the host.

2. Determine the size of each chunk.  
   "Small chunks" is vague, so I'll use a modest size such as 1024 bytes (1 KiB) for each allocation.  
   This size is large enough to be noticeable but small enough that 10 allocations won't consume much device memory.

3. Create an error‑checking macro to simplify CUDA API calls.  
   CUDA functions return a `cudaError_t`. If any call fails, the macro will print an error message and exit the program.

4. In `main()`:
   - Declare an array of 10 `void*` pointers on the host.
   - Loop from 0 to 9, calling `cudaMalloc(&ptrs[i], chunk_size)`.  
     On success, the pointer is stored in the array.
   - After the allocation loop, optionally print a confirmation message.
   - Loop again over the array, calling `cudaFree(ptrs[i])` for each pointer.
   - Print a final message indicating that all memory has been freed.

5. Add necessary includes (`<cstdio>`, `<cstdlib>`, `<cuda_runtime.h>`) and use `int main()`.

6. Ensure the program is self‑contained, compiles with `nvcc`, and does not require any external files or input.

With this plan, the resulting .cu file will allocate and free device memory correctly, handling errors and reporting progress to the console.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to check CUDA API call results
#define CHECK_CUDA_ERROR(call)                                            \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

int main() {
    const int NUM_CHUNKS = 10;
    const size_t CHUNK_SIZE = 1024; // 1 KiB per chunk

    // Allocate an array to hold device pointers
    void* dev_ptrs[NUM_CHUNKS];

    // Allocate device memory for each chunk
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        CHECK_CUDA_ERROR(cudaMalloc(&dev_ptrs[i], CHUNK_SIZE));
        printf("Allocated chunk %d at device address %p\n", i, dev_ptrs[i]);
    }

    // Optionally, do something with the allocated memory here
    // For this example, we simply free the memory.

    // Free each allocated chunk
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        CHECK_CUDA_ERROR(cudaFree(dev_ptrs[i]));
        printf("Freed chunk %d at device address %p\n", i, dev_ptrs[i]);
    }

    printf("All allocated memory has been freed successfully.\n");
    return 0;
}
