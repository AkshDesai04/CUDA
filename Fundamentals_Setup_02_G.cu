/*
Aim of the program: For each device, print its memory bus width in bits.
Thinking:
- Use CUDA runtime API to query devices.
- First, obtain the number of CUDA devices with cudaGetDeviceCount().
- Iterate over each device index.
- For each device, retrieve its properties using cudaGetDeviceProperties().
- The cudaDeviceProp structure contains a field called memoryBusWidth that stores the width of the memory bus in bits.
- Print the device index, device name, and the memory bus width.
- Use standard C++ headers (<cstdio>) for printing and <cuda_runtime.h> for CUDA functions.
- Basic error checking is performed after CUDA API calls to ensure robustness.
- The program compiles with nvcc and runs on any system with CUDA installed.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            return EXIT_FAILURE;                                     \
        }                                                             \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    }

    return 0;
}
