```cuda
/*
Aim: Allocate a very large amount of pinned memory (e.g., more than half your system RAM). Does it succeed? What happens to your system's responsiveness?

Thought Process:
- The task is to attempt to allocate pinned (page-locked) host memory that exceeds roughly 50% of the system's physical RAM.
- Pinned memory is allocated via cudaHostAlloc or cudaMallocHost. It is not a GPU memory allocation, so the size is limited by host OS constraints, not by the GPU's memory.
- On most systems, allocating such a large amount of pinned memory will either fail or cause the system to become unresponsive (swap thrashing, high memory pressure, etc.).
- We will:
  1. Determine the total physical RAM using sysconf() on POSIX systems.
  2. Compute half of that size.
  3. Attempt to allocate that amount with cudaHostAlloc (or cudaMallocHost).
  4. Check the return code and print whether the allocation succeeded.
  5. If it succeeded, keep the allocation for a while (e.g., 60 seconds) to allow observation of system responsiveness.
  6. Clean up and exit.
- For portability to non-POSIX systems (e.g., Windows), the code would need a different approach, but this example focuses on POSIX.
- The program is written in C/C++ with CUDA runtime API, and the entire code is within a single .cu file.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

int main(void) {
    // Determine total physical RAM (bytes)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages == -1 || page_size == -1) {
        perror("sysconf");
        return EXIT_FAILURE;
    }

    unsigned long long total_ram = (unsigned long long)pages * (unsigned long long)page_size;
    unsigned long long half_ram = total_ram / 2;

    printf("Total system RAM: %.2f GB\n", (double)total_ram / (1024.0 * 1024 * 1024));
    printf("Attempting to allocate pinned memory: %.2f GB\n", (double)half_ram / (1024.0 * 1024 * 1024));

    // Allocate pinned memory
    void* pinned_ptr = NULL;
    cudaError_t err = cudaHostAlloc(&pinned_ptr, half_ram, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        // Try cudaMallocHost as an alternative
        err = cudaMallocHost(&pinned_ptr, half_ram);
        if (err != cudaSuccess) {
            fprintf(stderr, "Pinned memory allocation failed: %s\n", cudaGetErrorString(err));
            printf("Result: Allocation FAILED.\n");
            return EXIT_FAILURE;
        } else {
            printf("Pinned memory allocation succeeded via cudaMallocHost.\n");
        }
    } else {
        printf("Pinned memory allocation succeeded via cudaHostAlloc.\n");
    }

    // Optional: Verify that we can write to the memory (to ensure it's usable)
    memset(pinned_ptr, 0, half_ram);
    printf("Pinned memory initialized.\n");

    // Keep the allocation for a while to observe system responsiveness
    printf("Sleeping for 60 seconds to observe system responsiveness...\n");
    fflush(stdout);
    sleep(60);

    // Clean up
    err = cudaFreeHost(pinned_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFreeHost failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Pinned memory freed successfully.\n");
    }

    printf("Done. Exiting.\n");
    return EXIT_SUCCESS;
}
```