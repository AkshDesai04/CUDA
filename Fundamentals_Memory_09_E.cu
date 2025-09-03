```cuda
/*
Aim: Allocate a very large amount of pinned memory (e.g., more than half your system RAM). Does it succeed? What happens to your system's responsiveness?

Thinking: Pinned (page‑locked) memory is allocated with cudaHostAlloc and is kept resident in physical RAM, preventing it from being swapped. Allocating an amount larger than the machine's available physical memory should cause the allocation to fail, or at least cause severe paging and degrade system responsiveness. To test this, the program queries the total system RAM using sysconf on Linux, then attempts to allocate roughly 1.5× that amount of pinned memory. It reports whether the allocation succeeded or failed, and frees the memory if it did succeed. This demonstrates the limits of pinned memory allocation and its impact on system performance.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>

int main(void)
{
    /* Get the total physical memory of the system in bytes */
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages == -1 || page_size == -1) {
        perror("sysconf");
        return EXIT_FAILURE;
    }
    size_t total_mem = (size_t)pages * (size_t)page_size;
    printf("Total physical memory detected: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));

    /* Attempt to allocate 1.5 times the total physical memory as pinned memory */
    size_t alloc_size = (size_t)(1.5 * (double)total_mem);
    printf("Attempting to allocate %zu bytes (~%.2f GB) of pinned memory...\n",
           alloc_size, alloc_size / (1024.0 * 1024.0 * 1024.0));

    void *pinned_mem = NULL;
    cudaError_t err = cudaHostAlloc(&pinned_mem, alloc_size, cudaHostAllocDefault);

    if (err == cudaSuccess) {
        printf("Pinned memory allocation succeeded.\n");
        /* Do something trivial to keep the memory alive for a short time */
        memset(pinned_mem, 0, alloc_size);
        /* Free the memory */
        cudaFreeHost(pinned_mem);
        printf("Pinned memory freed.\n");
    } else {
        printf("Pinned memory allocation failed with error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
```