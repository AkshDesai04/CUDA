```cpp
/*
Aim of the program:
Convert and print the clock rate in megahertz (MHz).

Thinking process:
- The goal is to query the GPU device clock rate and display it in MHz.
- CUDA exposes the device clock rate via cudaDeviceProp.clockRate, which is expressed in kilohertz (kHz).
- To convert to megahertz (MHz), divide the value by 1000.0.
- We'll query the properties of device 0 (the default device).
- The program will be a simple host-only application, no device kernels required.
- We'll use the CUDA Runtime API functions:
    * cudaGetDeviceProperties() to fetch the properties.
    * Standard C I/O (printf) to display the result.
- Error handling: check the return status of cudaGetDeviceProperties.
- Output will be a single line: "Device clock rate: X.YYY MHz".
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main()
{
    int device = 0;  // Default to device 0
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // prop.clockRate is in kHz. Convert to MHz.
    double clockRateMHz = static_cast<double>(prop.clockRate) / 1000.0;

    printf("Device clock rate: %.3f MHz\n", clockRateMHz);

    return EXIT_SUCCESS;
}
```