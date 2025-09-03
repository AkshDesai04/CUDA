```cu
/*
Aim: Write the memory-leaking code inside a function. Call this function in a loop in main to leak memory more quickly.

Thinking:
The task requires a CUDA program that intentionally leaks memory in order to demonstrate rapid memory exhaustion.
To achieve this, we will define a helper function `leakMemory()` that performs two allocations:
1. Host memory using `malloc`. This memory is never freed, causing a host-side leak.
2. Device memory using `cudaMalloc`. This memory is also never freed, causing a GPU-side leak.

The `main()` function will repeatedly call `leakMemory()` inside a loop (e.g., 1000 iterations), each time allocating 1 MB on both host and device.
Since the allocated buffers are never released, the program will consume increasing amounts of RAM and GPU memory.
We include basic error checking for the CUDA allocation but deliberately omit `free()` and `cudaFree()`.
After the loop, the program prints a completion message and exits, leaving the leaked memory in place.
This code can be compiled with `nvcc` and will demonstrate memory leaks on both host and device.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void leakMemory(void)
{
    const size_t MB = 1 << 20;          // 1 MB
    const size_t size = MB;             // allocation size

    /* Allocate host memory and intentionally leak it */
    char *hostPtr = (char *)malloc(size);
    if (hostPtr == NULL) {
        fprintf(stderr, "Host malloc failed\n");
        return;
    }
    /* No free(hostPtr); intentionally omitted */

    /* Allocate device memory and intentionally leak it */
    char *devPtr;
    cudaError_t err = cudaMalloc((void **)&devPtr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        /* hostPtr is still leaked; devPtr is not allocated */
        return;
    }
    /* No cudaFree(devPtr); intentionally omitted */

    /* Optionally, perform a trivial kernel launch or memory copy
       to keep the allocation in use. This is not required for the
       memory leak itself, but may help observe device memory usage. */
}

int main(void)
{
    const int iterations = 1000;
    for (int i = 0; i < iterations; ++i) {
        leakMemory();
        /* Optional: print progress to observe increasing usage */
        if ((i + 1) % 100 == 0) {
            printf("Leaked %d iterations\n", i + 1);
            fflush(stdout);
        }
    }

    printf("Completed %d memory leak iterations.\n", iterations);
    /* The program exits here; all leaked memory remains allocated. */
    return 0;
}
```