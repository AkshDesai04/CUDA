```cpp
/* 
Aim: 
Use `nvidia-smi` to monitor GPU memory. Run a program that allocates memory but doesn't free it. Then run a program that allocates memory and then calls `cudaDeviceReset()`. Observe how `nvidia-smi` shows the memory being reclaimed in the second case.

Thinking:
- We want a single .cu file that can be built with `nvcc`. 
- The program will accept an optional command line argument: "reset". 
  - If "reset" is provided, the program will allocate a chunk of device memory, then call `cudaDeviceReset()` to force the driver to clean up allocations. After the reset, the program exits.
  - If no argument is provided, the program will allocate the same amount of device memory but will not free it explicitly. Instead it will sleep for a while (e.g., 60 seconds) so that the user can observe the memory usage with `nvidia-smi`. The program will exit after the sleep; the driver will clean up automatically when the process ends, but this happens later than the explicit `cudaDeviceReset`.
- We allocate a modest amount of memory (256 MiB) to keep the example quick while still visible in `nvidia-smi`.
- Error checking is performed after each CUDA API call to make debugging easier.
- The code uses only CUDA runtime API and standard C++/C headers, so it can be compiled with a single `nvcc` invocation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(int argc, char* argv[])
{
    const size_t allocSize = 256 * 1024 * 1024; // 256 MiB
    void* d_ptr = NULL;

    printf("Allocating %zu MiB of device memory...\n", allocSize / (1024 * 1024));
    CUDA_CHECK(cudaMalloc(&d_ptr, allocSize));
    printf("Allocation succeeded.\n");

    if (argc > 1 && strcmp(argv[1], "reset") == 0) {
        printf("Calling cudaDeviceReset() to free allocations immediately.\n");
        CUDA_CHECK(cudaDeviceReset());
        printf("cudaDeviceReset() completed. Device memory should be reclaimed.\n");
        // Note: After cudaDeviceReset, the device context is destroyed, so
        // we cannot use d_ptr after this point. The program exits now.
        return EXIT_SUCCESS;
    }

    printf("No reset requested. Sleeping for 60 seconds to allow observation with nvidia-smi.\n");
    printf("Press Ctrl+C to exit early.\n");
    sleep(60);  // Keep the program alive for observation

    // Optionally free the memory before exiting (not strictly necessary for this demo)
    CUDA_CHECK(cudaFree(d_ptr));
    printf("Freed device memory and exiting.\n");

    return EXIT_SUCCESS;
}
```