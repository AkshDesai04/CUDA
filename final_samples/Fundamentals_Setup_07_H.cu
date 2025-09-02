```cpp
/*
Aim of the program: Check if the device supports cooperative launch (`cooperativeLaunch`) and print the result.

Thought process:
1. The goal is to query the CUDA device to determine whether it supports cooperative launch, a feature that allows multiple kernels to cooperate on a single device launch.
2. CUDA provides the device property `cooperativeLaunch` within the `cudaDeviceProp` structure, which is set to 1 if the device supports cooperative launches.
3. We will:
   a. Determine the number of CUDA-capable devices.
   b. If at least one device exists, query the properties of device 0.
   c. Examine the `cooperativeLaunch` field and print whether cooperative launch is supported.
4. The program uses standard CUDA runtime API calls:
   - `cudaGetDeviceCount`
   - `cudaGetDeviceProperties`
   - `cudaGetErrorString` for error reporting.
5. The output will be:
   - "Device 0 supports cooperative launch: Yes" or
   - "Device 0 supports cooperative launch: No"
   - If no devices are present or an error occurs, an appropriate message will be printed.
6. The code is written in C++ but compatible with CUDA C, and is self-contained in a single .cu file.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    int device = 0; // Check the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // The cooperativeLaunch field indicates if the device supports cooperative launches.
    bool supportsCooperativeLaunch = prop.cooperativeLaunch != 0;

    printf("Device %d supports cooperative launch: %s\n", device,
           supportsCooperativeLaunch ? "Yes" : "No");

    return EXIT_SUCCESS;
}
```