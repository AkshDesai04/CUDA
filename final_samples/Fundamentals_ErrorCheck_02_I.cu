/*
Go back to the device query programs from `Fundamentals_Setup` and add `CHECK` macros to all API calls.

Thinking:
- The original device query sample uses CUDA runtime API calls such as cudaGetDeviceCount, cudaGetDeviceProperties, etc., without explicit error handling.
- To enforce robust error checking, we define a macro CHECK that wraps each API call, checks the returned cudaError_t, prints an informative message if an error occurs, and aborts the program.
- The macro uses cudaGetErrorString to obtain a readable error description, and prints the file and line number via __FILE__ and __LINE__ macros.
- We then replace all direct CUDA API calls in the program with the CHECK macro, ensuring that no call is left unchecked.
- The program queries the number of CUDA-capable devices, then iterates over each device, obtains its properties, and prints key information (name, compute capability, total global memory, max threads per block, and the number of multiprocessors).
- The code is written in C style and compiled with nvcc, producing a .cu file. No external dependencies beyond the CUDA runtime are required.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls and abort on error */
#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    /* Get number of devices */
    CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    printf("Detected %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        /* Get device properties */
        CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("\nDevice %d: \"%s\"\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu bytes (%.2f GB)\n",
               prop.totalGlobalMem,
               (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Register per block: %d\n", prop.regsPerBlock);
    }

    return EXIT_SUCCESS;
}
