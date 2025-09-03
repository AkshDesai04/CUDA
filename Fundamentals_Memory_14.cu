```cuda
/*
Aim of the program:
Allocate a device array. Pass its pointer to a kernel. Inside the kernel, deliberately write to an out-of-bounds index (e.g., d_ptr[BIG_NUMBER]).

Thinking process:
1. We need a simple CUDA program that allocates a small array on the device.
2. The array will be of known size (e.g., 10 elements) so that writing far beyond this size will trigger a memory error.
3. A macro BIG_NUMBER will be set to a large value (e.g., 1000) that is definitely outside the bounds of the allocated array.
4. The kernel will receive the pointer and write to d_ptr[BIG_NUMBER]. This access is out of bounds and should be caught by cuda-memcheck.
5. In the host code, we will:
   - Allocate the device array with cudaMalloc.
   - Launch the kernel with a single thread (since the out-of-bounds write does not depend on thread indexing).
   - Synchronize and check for errors.
   - Free the device memory.
6. The program will compile with nvcc and be run under cuda-memcheck, which will report the invalid write.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Size of the allocated array on device
#define ARRAY_SIZE 10
// A large index that is definitely out of bounds
#define BIG_NUMBER 1000

// Kernel that writes to an out-of-bounds index
__global__ void out_of_bounds_kernel(int *d_ptr)
{
    // Deliberate out-of-bounds write
    d_ptr[BIG_NUMBER] = 42;
}

int main()
{
    int *d_array = NULL;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void **)&d_array, ARRAY_SIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Launch kernel with one thread
    out_of_bounds_kernel<<<1, 1>>>(d_array);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch any runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_array);

    printf("Program finished. Run with cuda-memcheck to see the out-of-bounds write.\n");
    return EXIT_SUCCESS;
}
```