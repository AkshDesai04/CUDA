```cpp
/*
Allocate a device array. Pass its pointer to a kernel. Inside the kernel, deliberately write to an out-of-bounds index (e.g., `d_ptr[BIG_NUMBER]`). Run the program using `cuda-memcheck` and analyze its report to find the memory error.

Thinking process:
1. We need a minimal CUDA program that allocates a small array on the device.
2. The kernel will receive this pointer and will write to an index beyond the allocated size.
   Choosing a constant BIG_NUMBER that is well beyond the array size guarantees an OOB access.
3. The program will launch the kernel, synchronize, and then exit.
4. We include basic CUDA error checking to report any launch failures.
5. Users can compile with `nvcc` and run under `cuda-memcheck` to see the reported out-of-bounds write.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 10          // Size of the allocated array
#define BIG_NUMBER 100         // Index deliberately out of bounds

// Simple error-checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel that writes to an out-of-bounds index
__global__ void outOfBoundsKernel(int *d_ptr) {
    // Every thread writes to the same out-of-bounds index
    d_ptr[BIG_NUMBER] = 42;  // This access is out of bounds
}

int main() {
    int *d_array = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_array, ARRAY_SIZE * sizeof(int)));

    // Optionally initialize the array to zero
    CUDA_CHECK(cudaMemset(d_array, 0, ARRAY_SIZE * sizeof(int)));

    // Launch kernel with a single block and thread
    outOfBoundsKernel<<<1, 1>>>(d_array);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure kernel completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK(cudaFree(d_array));

    printf("Program completed. Run with cuda-memcheck to see the out-of-bounds write.\n");
    return 0;
}
```