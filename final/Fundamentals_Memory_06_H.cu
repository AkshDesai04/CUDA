```cuda
/*
Aim: In a loop, allocate memory and overwrite the pointer variable in the next iteration. Explain why this is a leak.

Thinking:
- In CUDA, cudaMalloc allocates memory on the device and returns a device pointer.
- If we keep allocating inside a loop but keep reassigning the pointer variable without freeing the previous allocation, the previous allocations are lost and cannot be freed.
- This results in a memory leak on the GPU device, consuming device memory until exhaustion.
- The last allocation can be freed, but all previous ones are unrecoverable because we lost their addresses.
- Example code demonstrates this: loop with cudaMalloc, no cudaFree inside loop.
- After loop, we free only the last allocation.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    const int numIterations = 10;
    const size_t allocSize = 1 << 20; // 1 MB per allocation

    // Device pointer that will be overwritten each iteration
    int *d_ptr = nullptr;

    printf("Allocating memory %d times without freeing previous allocations.\n", numIterations);

    for (int i = 0; i < numIterations; ++i)
    {
        cudaError_t err = cudaMalloc((void**)&d_ptr, allocSize);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc failed at iteration %d: %s\n", i, cudaGetErrorString(err));
            // Attempt to clean up before exiting
            if (d_ptr) cudaFree(d_ptr);
            return EXIT_FAILURE;
        }

        // Normally we would use d_ptr here, but we immediately overwrite it
        // in the next iteration without freeing the previous allocation.
    }

    // At this point, all allocations except the last one are leaked.
    // We can free only the last allocation because we still hold its address.
    if (d_ptr)
    {
        cudaError_t err = cudaFree(d_ptr);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
            return EXIT_FAILURE;
        }
    }

    printf("Only the last allocation was freed. The previous %d allocations are leaked.\n", numIterations - 1);
    return EXIT_SUCCESS;
}
```