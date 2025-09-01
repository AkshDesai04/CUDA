```cpp
/*
Aim of the program: Convert and print the clock rate in gigahertz (GHz).

Thought process:
1. The CUDA API provides the device clock rate in the `cudaDeviceProp.clockRate` field, measured in kilohertz (kHz).
2. To express this in gigahertz (GHz), we divide the kHz value by 1,000,000 (since 1 GHz = 1,000,000 kHz).
3. The program will iterate over all CUDA-capable devices available on the system, query their properties, perform the conversion, and print the result.
4. Basic error handling will be included to capture CUDA API failures.
5. The output will be formatted to display the device index, name, and clock rate in GHz with three decimal places.
6. The code is self-contained in a single `.cu` file and can be compiled with `nvcc`.

*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        // prop.clockRate is in kHz; convert to GHz
        double clockGHz = static_cast<double>(prop.clockRate) / 1e6;

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Clock Rate: %.3f GHz\n", clockGHz);
    }

    return 0;
}
```