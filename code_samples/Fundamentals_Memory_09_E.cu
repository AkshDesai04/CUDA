/*
 * Aim: Allocate a very large amount of pinned memory (e.g., more than half your system RAM).
 * Does it succeed? What happens to your system's responsiveness?
 *
 * Thinking:
 * 1. Pinned (page-locked) host memory is allocated using cudaMallocHost.
 *    It forces the OS to keep the pages resident in RAM, preventing them from being swapped out.
 * 2. If we request a size larger than the available physical memory, the allocation may still succeed
 *    because the OS can commit virtual pages, but the pages will be backed by swap or may fail outright
 *    depending on the OS's memory management policies.
 * 3. Allocating a huge amount of pinned memory reduces the amount of RAM available for other processes
 *    and can cause the system to become sluggish or start swapping other memory pages, degrading responsiveness.
 * 4. This program attempts to allocate 16 GB of pinned memory (on systems with 32 GB RAM or more).
 *    It prints whether the allocation succeeded, displays the total physical RAM, and sleeps for a
 *    short period while the allocation remains in place so you can observe any system slowdown.
 * 5. The program then frees the memory and exits.
 *    Note: If your system has less than 16 GB of RAM, adjust SIZE_GB accordingly.
 * 
 * Usage:
 *   Compile with: nvcc -o pinned_alloc pinned_alloc.cu
 *   Run: ./pinned_alloc
 *
 * The program will print diagnostic messages to stdout.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

#define SIZE_GB 16ULL   // Desired pinned memory size in gigabytes
#define SLEEP_SECONDS 10 // How long to keep the allocation alive

int main(void)
{
    // Determine total physical RAM
    struct sysinfo sInfo;
    if (sysinfo(&sInfo) != 0) {
        perror("sysinfo");
        return EXIT_FAILURE;
    }
    unsigned long long total_physical_mem = (unsigned long long)sInfo.totalram * sInfo.mem_unit;
    printf("Total physical RAM: %.2f GB\n", (double)total_physical_mem / (1024 * 1024 * 1024));

    // Calculate requested allocation size
    unsigned long long alloc_size = SIZE_GB * 1024ULL * 1024ULL * 1024ULL;
    printf("Attempting to allocate %.2f GB of pinned host memory...\n", (double)alloc_size / (1024 * 1024 * 1024));

    // Attempt to allocate pinned memory
    void *pinned_ptr = NULL;
    cudaError_t err = cudaMallocHost(&pinned_ptr, alloc_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Pinned memory allocation succeeded.\n");
    printf("You may observe system responsiveness changes for the next %d seconds.\n", SLEEP_SECONDS);

    // Touch the memory to ensure pages are faulted in
    // Use memset for simplicity; this may also help pin the pages in RAM
    memset(pinned_ptr, 0, (size_t)alloc_size);

    // Sleep for the specified duration
    sleep(SLEEP_SECONDS);

    // Free the pinned memory
    err = cudaFreeHost(pinned_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFreeHost failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Pinned memory freed. Program exiting normally.\n");
    return EXIT_SUCCESS;
}
