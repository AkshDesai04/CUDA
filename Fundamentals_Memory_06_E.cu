```c
/*
 * Use `cudaMemGetInfo` to query the amount of free and total memory on the device.
 * Print these values before and after the deliberate leak.
 *
 * ----- Thinking Process -----
 * The goal is to demonstrate how `cudaMemGetInfo` reflects GPU memory usage changes.
 * We'll start by querying the free and total memory, then deliberately allocate a
 * chunk of GPU memory with `cudaMalloc` and **not** free it.  After the allocation
 * we query the memory info again to see the reduced free memory.  This is a simple
 * self‑contained example that can be compiled with `nvcc`.
 *
 * Steps:
 * 1. Include necessary headers (`stdio.h`, `cuda_runtime.h`).
 * 2. In `main`, declare `size_t free, total;` and call `cudaMemGetInfo` to get
 *    initial values.  Print them using `%zu` for `size_t`.
 * 3. Allocate a reasonably sized block (e.g., 100 MiB).  Check for allocation
 *    success; if it fails, print an error and exit.
 * 4. Do NOT call `cudaFree` – this is the intentional leak.
 * 5. Call `cudaMemGetInfo` again and print the new values.
 * 6. Return 0.
 *
 * Note: The program purposely leaks GPU memory.  In a real application you
 * should always free allocated memory.  The output will show the decrease
 * in free memory after the allocation.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    size_t freeMem, totalMem;
    cudaError_t err;

    /* Query memory info before allocation */
    err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed before allocation: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    printf("Before leak:  Free = %zu bytes, Total = %zu bytes\n",
           freeMem, totalMem);

    /* Allocate a block of memory (100 MiB) to create a deliberate leak */
    size_t leakSize = 100ULL * 1024ULL * 1024ULL; /* 100 MiB */
    void *d_ptr = NULL;
    err = cudaMalloc(&d_ptr, leakSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Allocated %zu bytes on the device (leak)\n", leakSize);

    /* Query memory info after allocation (leak) */
    err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed after allocation: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    printf("After leak:   Free = %zu bytes, Total = %zu bytes\n",
           freeMem, totalMem);

    /* Intentionally do NOT free the allocated memory to create a leak */

    return 0;
}
```