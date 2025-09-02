```cuda
/*

Write a program that queries the available global memory and prints a warning if it is less than a certain threshold (e.g., 4096 MB).

Thought process:
- The CUDA runtime provides cudaMemGetInfo(&free, &total) which returns the amount of free memory and the total memory on the current device.
- Both values are returned as size_t (usually 64‑bit), so they can comfortably hold the memory amounts on modern GPUs.
- Convert the values from bytes to megabytes (MB) by dividing by (1024 * 1024).
- Define a threshold in MB (4096 MB in this example).
- If the free memory is less than the threshold, print a warning message; otherwise, print a normal informational message.
- Always check the CUDA error code returned by cudaMemGetInfo and exit with a non‑zero status if an error occurs.
- The program uses only standard C/C++ headers and CUDA runtime headers, so it can be compiled with nvcc.

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(void) {
    size_t free_mem_bytes = 0;
    size_t total_mem_bytes = 0;

    cudaError_t err = cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    const size_t bytes_per_mb = 1024 * 1024;
    size_t free_mem_mb = free_mem_bytes / bytes_per_mb;
    size_t total_mem_mb = total_mem_bytes / bytes_per_mb;

    printf("Total GPU memory: %zu MB\n", total_mem_mb);
    printf("Free GPU memory:  %zu MB\n", free_mem_mb);

    const size_t threshold_mb = 4096;  // Threshold in MB

    if (free_mem_mb < threshold_mb) {
        fprintf(stderr, "WARNING: Available global memory (%zu MB) is below the threshold of %zu MB.\n",
                free_mem_mb, threshold_mb);
        return EXIT_FAILURE;
    }

    printf("Available global memory is above the threshold.\n");
    return EXIT_SUCCESS;
}
```