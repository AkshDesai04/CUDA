```cpp
/*
Allocate a very large amount of pinned memory (e.g., more than half your system RAM). Does it succeed? What happens to your system's responsiveness?

I decided to write a CUDA C program that tries to allocate a huge amount of pinned host memory using cudaMallocHost. The idea is to request a size that is larger than the available system RAM. Since pinned memory is backed by physical RAM, requesting more than the machine can supply should cause the allocation to fail with an error code from the CUDA runtime. 

On most systems, attempting to pin such a large amount of memory will either immediately fail or, if the system is under heavy load, may cause a noticeable slowdown or even unresponsiveness because the OS has to move pages around and cannot page out the pinned region. In some very large allocations, the OS might also trigger a memory pressure response, potentially swapping or killing processes. 

To make the test meaningful across platforms, I will:
1. Query the total physical RAM using sysconf on Linux (on other OSes this can be adapted).
2. Compute half of that amount.
3. Try to allocate that many bytes with cudaMallocHost.
4. Check the return status. If it succeeds, I will print a success message and then free the memory. If it fails, I will print the CUDA error string.
5. Additionally, I will print the current available free device memory using cudaMemGetInfo for reference, although device memory is unrelated to pinned host allocation.

The program will be a minimal .cu file that can be compiled with nvcc. It should compile and run on any system with CUDA installed, and it will attempt the allocation and report the outcome.

*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

/* Helper to print CUDA errors */
static void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Get total physical RAM in bytes (Linux implementation) */
static size_t getTotalPhysicalRAM()
{
#if defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages == -1 || page_size == -1)
    {
        fprintf(stderr, "Error obtaining physical memory info.\n");
        exit(EXIT_FAILURE);
    }
    return (size_t)pages * (size_t)page_size;
#else
    /* On non-Linux systems, fallback to 8GB as a guess */
    return (size_t)8ULL * 1024ULL * 1024ULL * 1024ULL;
#endif
}

int main()
{
    /* Query total system RAM */
    size_t totalRAM = getTotalPhysicalRAM();
    printf("Total physical RAM: %.2f GB\n", (double)totalRAM / (1024.0 * 1024.0 * 1024.0));

    /* Attempt to allocate slightly more than half of total RAM */
    size_t requestedSize = totalRAM / 2 + (size_t)(128ULL * 1024ULL * 1024ULL); /* +128 MB */
    printf("Requesting pinned host memory: %.2f GB\n", (double)requestedSize / (1024.0 * 1024.0 * 1024.0));

    void *pinnedMem = NULL;
    cudaError_t err = cudaMallocHost(&pinnedMem, requestedSize);

    if (err == cudaSuccess)
    {
        printf("Pinned memory allocation succeeded.\n");
        /* Optionally do something with the memory */
        memset(pinnedMem, 0, requestedSize);
    }
    else
    {
        printf("Pinned memory allocation FAILED: %s\n", cudaGetErrorString(err));
    }

    /* Print device memory status for reference */
    size_t freeDevice, totalDevice;
    cudaMemGetInfo(&freeDevice, &totalDevice);
    printf("Device memory: %.2f GB free / %.2f GB total\n",
           (double)freeDevice / (1024.0 * 1024.0 * 1024.0),
           (double)totalDevice / (1024.0 * 1024.0 * 1024.0));

    /* Free pinned memory if allocated */
    if (pinnedMem)
    {
        cudaFreeHost(pinnedMem);
        printf("Pinned memory freed.\n");
    }

    return 0;
}
```