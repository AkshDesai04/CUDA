/*
Go back to the device query programs from `Fundamentals_Setup` and add `CHECK` macros to all API calls.

Thinking:
1. The original device query program enumerates CUDA-capable devices and prints properties.
2. The user requests adding CHECK macros around all API calls to ensure proper error handling.
3. Define a macro `CHECK` that executes a CUDA runtime API call, captures its error code,
   and if the error is not `cudaSuccess`, prints a detailed message (including the call
   string, file, line, and the CUDA error string) and exits the program.
4. Wrap every `cuda*` call in the program with this macro: `cudaGetDeviceCount`, 
   `cudaGetDeviceProperties`, `cudaGetLastError` (if used), `cudaDeviceReset`, etc.
5. Use standard headers: `<cuda_runtime.h>`, `<stdio.h>`, `<stdlib.h>`.
6. The main function:
   - Retrieve device count.
   - Loop over each device, retrieve its properties, and print key details:
     name, compute capability, total global memory, shared memory per block,
     registers per block, warp size, max threads per block, etc.
   - At the end, optionally reset the device.
7. Ensure the code compiles with `nvcc` and produces a single `.cu` file containing the
   comment and the full program. No extra output or explanations outside the file.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Macro to check CUDA runtime API calls.
   If an error occurs, print details and exit. */
#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable device detected.\n");
        return EXIT_SUCCESS;
    }

    printf("Detected %d CUDA-capable device(s):\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("\nDevice %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n",
               (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %.2f KB\n",
               (double)prop.sharedMemPerBlock / 1024.0);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("  Clock Rate: %.2f MHz\n", (double)prop.clockRate / 1000.0);
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    }

    /* Reset the device (optional, but good practice for cleanup) */
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
