/*
Aim of the program: Convert and print the clock rate in megahertz (MHz).

The program is intended to query the CUDA-capable GPUs present in the system, retrieve the
device properties via the CUDA Runtime API, and then convert the reported clock rate
from the native unit (kHz) to megahertz (MHz).  The CUDA device property `clockRate`
returns the core clock frequency in kilohertz, so a simple division by 1000.0
produces the frequency in megahertz.

Key steps implemented:
1. Call `cudaGetDeviceCount` to find the number of CUDA-capable devices.
2. For each device:
   a. Retrieve `cudaDeviceProp` using `cudaGetDeviceProperties`.
   b. Extract `clockRate` (kHz).
   c. Convert to MHz: `double MHz = clockRate / 1000.0`.
   d. Print the device index, name, and clock rate in MHz.
3. Simple error handling: check return status of CUDA calls and print error messages
   if anything fails.  This ensures the program fails gracefully if the device
   queries are unavailable or the system has no CUDA GPUs.

The program is written in C, compiled as a CUDA source file (.cu) and can be built
using `nvcc`.  The output will look like:

```
Number of CUDA devices: 1
GPU 0: Tesla V100: Clock rate: 1398.0 MHz
```

All required headers are included, and the program is self‑contained.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                                  \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                         \
        }                                                                \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        /* clockRate is in kHz, convert to MHz */
        double MHz = prop.clockRate / 1000.0;

        printf("GPU %d: %s: Clock rate: %.3f MHz\n",
               dev, prop.name, MHz);
    }

    return EXIT_SUCCESS;
}
