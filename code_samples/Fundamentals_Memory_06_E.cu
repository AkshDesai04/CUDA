```c
/*
Aim of the program:
Use `cudaMemGetInfo` to query the amount of free and total memory on the device.
Print these values before and after the deliberate leak.

Thought process:
1. The program must be a CUDA C source file (`.cu`) that uses the CUDA runtime API.
2. We need to call `cudaMemGetInfo` before any allocation to get the initial free/total memory.
3. Then we deliberately allocate some device memory with `cudaMalloc` but intentionally
   do not free it. This creates a memory leak on the GPU.
4. After the allocation we call `cudaMemGetInfo` again to show the effect of the leak.
5. All output is printed to the console using `printf`.
6. Basic error checking is added after each CUDA call; if an error occurs, we print the
   error string and exit.
7. The program is self‑contained and uses only the CUDA runtime header `<cuda_runtime.h>`
   and the standard C header `<stdio.h>`. It is written in pure C (no C++ features).
8. The code is compiled with `nvcc` (e.g., `nvcc -o memleak memleak.cu`) and runs on any
   device with CUDA support.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper function to print free and total memory on the current device. */
static void print_mem_info(const char *msg)
{
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: Failed to get memory info: %s\n", msg, cudaGetErrorString(err));
        return;
    }
    printf("%s: Free memory = %zu bytes, Total memory = %zu bytes\n",
           msg, free_mem, total_mem);
}

int main(void)
{
    /* Print memory info before allocation */
    print_mem_info("Before allocation");

    /* Allocate 256 MB of device memory intentionally not freed */
    const size_t alloc_size = 256 * 1024 * 1024; /* 256 MB */
    void *d_ptr = NULL;
    cudaError_t err = cudaMalloc(&d_ptr, alloc_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate %zu bytes on device: %s\n",
                alloc_size, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    /* Print memory info after allocation (leak) */
    print_mem_info("After allocation (leak)");

    /* Intentionally do not free d_ptr to create a leak */

    /* Optional: sync device to ensure allocation is processed */
    cudaDeviceSynchronize();

    /* Print memory info again after synchronizing */
    print_mem_info("After cudaDeviceSynchronize (leak still present)");

    /* Normally we would free d_ptr here, but we omit that to keep the leak. */

    return EXIT_SUCCESS;
}
```