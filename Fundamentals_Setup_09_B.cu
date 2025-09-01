/*
Aim: Query and print the `asyncEngineCount` property, which indicates the number of engines available for overlapping data copies and kernel execution.

Thinking:
1. The CUDA runtime provides device properties via the `cudaDeviceProp` struct, accessible through `cudaGetDeviceProperties`. The `asyncEngineCount` field inside this struct holds the number of asynchronous engines on a GPU.
2. The program will:
   - Retrieve the number of CUDA-capable devices with `cudaGetDeviceCount`.
   - Iterate over each device, query its properties, and print the `asyncEngineCount`.
3. To keep the code robust, a simple error-checking macro (`CHECK_CUDA`) will wrap CUDA API calls and report any failures.
4. The code will be compiled with `nvcc` and run on systems with CUDA installed. It will output the async engine count for each detected device.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: asyncEngineCount = %d\n",
               dev, prop.asyncEngineCount);
    }

    return EXIT_SUCCESS;
}
