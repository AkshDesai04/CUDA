```cpp
/* 
Try passing a pointer from `cudaMallocHost` to `free()` instead of `cudaFreeHost`. What happens?

The aim of this program is to demonstrate the consequences of freeing host pinned memory
that was allocated with `cudaMallocHost` using the standard C library function `free()`
instead of CUDA's `cudaFreeHost()`. `cudaMallocHost` allocates pinned (page-locked) memory
which is managed by the CUDA runtime. The memory block is not part of the standard
heap that `free()` expects to deallocate. Therefore, calling `free()` on such a pointer
results in undefined behavior: the program may crash with a segmentation fault, corrupt
the heap, or appear to work incorrectly. This program allocates a small array of
integers, writes a value, attempts to free it with `free()`, and then prints a message
indicating whether the program survived. In practice, the program typically aborts
or crashes, demonstrating that `free()` should not be used for memory allocated
by `cudaMallocHost`. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void)
{
    const size_t N = 10;
    int *p = NULL;

    // Allocate pinned host memory
    cudaError_t err = cudaMallocHost((void**)&p, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Use the memory
    for (size_t i = 0; i < N; ++i) {
        p[i] = (int)i;
    }

    printf("First element before free: %d\n", p[0]);

    // Attempt to free pinned memory with standard free()
    // This is incorrect and leads to undefined behavior.
    free(p);

    // Attempt to print after free (may crash or print old value)
    // We guard this with a check to see if the program hasn't crashed.
    // In practice, the program may have already terminated.
    printf("Attempting to print after free: %d\n", p[0]);

    // Reset CUDA device to clean up
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
```