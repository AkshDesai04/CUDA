```c
/*
Iterate through all devices on the system and print the asyncEngineCount for each.

The goal of this program is to enumerate every CUDA-capable device available on the host machine
and report the number of asynchronous engine units (asyncEngineCount) each device possesses.
These engine units are used internally by the CUDA driver to handle concurrent kernel
execution and memory transfers.  Knowing their count can help with performance tuning
for applications that rely on overlapping work.

Approach:
1. Use cudaGetDeviceCount() to obtain the number of CUDA devices.
2. Loop over each device index.
3. For each device:
   - Set the device as the active device (optional but good practice).
   - Query its properties via cudaGetDeviceProperties().
   - Retrieve prop.asyncEngineCount.
   - Print the device index and asyncEngineCount.
4. Include basic error handling to report any CUDA API failures.

The program uses the CUDA Runtime API, is written in plain C, and can be compiled
with `nvcc -o device_async_engine_count device_async_engine_count.cu`.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            return EXIT_FAILURE;                                 \
        }                                                         \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        // Set the current device (not strictly required for querying properties)
        CHECK_CUDA(cudaSetDevice(dev));

        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  asyncEngineCount = %d\n", prop.asyncEngineCount);
    }

    return EXIT_SUCCESS;
}
```