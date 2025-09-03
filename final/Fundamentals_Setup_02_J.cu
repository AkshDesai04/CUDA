```cpp
/*
Aim of the program: For each device, print its maximum grid dimensions ([X, Y, Z]).

Thinking:
- We need a CUDA C program (.cu) that queries all available CUDA devices.
- Use cudaGetDeviceCount to get the number of devices.
- Loop over each device index.
- For each device, set it as the current device using cudaSetDevice (optional but good practice).
- Retrieve its properties with cudaGetDeviceProperties.
- The maximum grid dimensions are stored in deviceProp.maxGridSize[3].
- Print them in a readable format: "Device <index>: max grid dimensions = [X, Y, Z]".
- Include error checking for CUDA API calls to ensure robustness.
- The program should be self-contained, compile with nvcc, and run on any system with CUDA installed.
- Output will be printed to stdout.
- No additional functionality is needed; we just display the requested information.
*/
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        // Optional: set the device as current
        CUDA_CHECK(cudaSetDevice(dev));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d (%s): max grid dimensions = [%d, %d, %d]\n",
               dev,
               prop.name,
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
    }

    return EXIT_SUCCESS;
}
```