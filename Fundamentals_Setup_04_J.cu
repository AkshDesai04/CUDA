/*
Aim: Query and print whether the device can map host memory (canMapHostMemory property).

To achieve this, we:
1. Include the necessary CUDA runtime header and standard IO.
2. Define a helper macro for error checking after CUDA API calls.
3. In the main function, obtain the number of CUDA devices available.
4. Iterate over each device, retrieving its properties with `cudaGetDeviceProperties`.
5. Print the device index and the value of the `canMapHostMemory` property, which is an integer (0 or 1).
6. Handle any errors that may arise during the property query.
7. Exit with status 0 on success.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA API errors
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            return 1;                                            \
        }                                                         \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("  canMapHostMemory: %s\n",
               prop.canMapHostMemory ? "Yes" : "No");
    }

    return 0;
}
