```cuda
/*
Aim: Convert and print the clock rate in megahertz (MHz).

Thinking:
1. The user requests a CUDA program in C that converts and prints the clock rate in MHz.
2. In CUDA, device properties can be obtained with cudaGetDeviceProperties().
3. The field `clockRate` in cudaDeviceProp is expressed in kilohertz (kHz).
4. To convert to megahertz (MHz), divide the kHz value by 1000.0.
5. The program will:
   - Query the number of CUDA-capable devices with cudaGetDeviceCount().
   - For each device, retrieve its properties.
   - Convert the clockRate to MHz and print the device name along with the converted rate.
6. Include error checking after CUDA API calls to ensure robustness.
7. The code is written in C syntax (no C++ features) and is saved in a .cu file for compilation with nvcc.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Utility macro for checking CUDA API call results */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            return EXIT_FAILURE;                                      \
        }                                                             \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        /* prop.clockRate is in kilohertz (kHz). Convert to megahertz (MHz). */
        double clockMHz = prop.clockRate / 1000.0;

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Clock Rate: %.2f MHz\n\n", clockMHz);
    }

    return EXIT_SUCCESS;
}
```