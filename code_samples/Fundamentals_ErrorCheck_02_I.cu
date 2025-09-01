```cpp
/*
Go back to the device query programs from Fundamentals_Setup and add CHECK macros to all API calls

In this task, I need to produce a CUDA device query program that follows the style of the examples in the
“Fundamentals_Setup” directory. The key requirement is to wrap every CUDA Runtime API call in a
CHECK macro so that errors are caught immediately and reported with useful context. The macro should
take the CUDA function call as an argument, execute it, check the returned cudaError_t, and if
the result is not cudaSuccess, print a helpful message containing the file, line, error string,
and the failed call itself, then abort the program.

I will use a typical device query pattern:
1. Get the number of CUDA-capable devices with cudaGetDeviceCount.
2. Loop over each device, retrieving its properties with cudaGetDeviceProperties.
3. Print out selected properties such as device name, compute capability, total global memory,
   and number of multiprocessors.
4. Finally reset the device with cudaDeviceReset.

All calls to the CUDA runtime will be wrapped with the CHECK macro. The program will include
the necessary headers, define the macro, and implement a main function that performs the
query and outputs information to stdout. The program will be a single .cu file with no
additional comments or output.

The resulting code is below.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro to check CUDA API calls and abort on error
#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err), #call);\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA.\n");
        return 0;
    }

    printf("CUDA Device Query (Runtime API) - Result: PASS\n");
    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("\nDevice %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n",
               prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n",
               static_cast<double>(prop.totalGlobalMem) / (1024 * 1024 * 1024));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  CUDA Cores per MP: %d\n", prop.multiProcessorCount); // simplified
        printf("  Clock Rate: %d kHz\n", prop.clockRate);
    }

    // Reset device(s)
    for (int dev = 0; dev < deviceCount; ++dev) {
        CHECK(cudaSetDevice(dev));
        CHECK(cudaDeviceReset());
    }

    return 0;
}
```