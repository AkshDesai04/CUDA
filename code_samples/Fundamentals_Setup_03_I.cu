```cpp
/* 
Calculate and print the theoretical peak memory bandwidth of the primary device in GB/s. 
Formula: `(memoryClockRate_kHz * 1000 * memoryBusWidth_bits / 8) / 1e9 * 2` (the *2 is for DDR memory).

Thinking process:
1. We need to use the CUDA Runtime API to query device properties. The primary device is typically device 0, but we first set the device to the current device or just query device 0.
2. The relevant properties are:
   - `memoryClockRate`: the memory clock rate in kHz.
   - `memoryBusWidth`: the memory bus width in bits.
3. Compute the theoretical peak bandwidth using the given formula:
   ```
   double bandwidth = (memoryClockRate * 1000.0 * memoryBusWidth / 8.0) / 1e9 * 2.0;
   ```
   Explanation:
   - `memoryClockRate` is in kHz, so we multiply by 1000 to get Hz.
   - Multiply by bus width in bits and divide by 8 to convert to bytes.
   - Divide by 1e9 to convert bytes per second to gigabytes per second.
   - Multiply by 2 because DDR memory transfers twice per cycle.
4. Print the result in GB/s with a reasonable precision.
5. Add error checking for CUDA calls. If no device is found or a CUDA error occurs, report and exit gracefully.
6. The code is a self-contained CUDA program in a single .cu file, no external dependencies beyond the CUDA runtime.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // We'll use the first device (device 0) as the primary device.
    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Compute theoretical peak memory bandwidth.
    double memoryClockRateHz = prop.memoryClockRate * 1000.0; // kHz to Hz
    double busWidthBits = (double)prop.memoryBusWidth;
    double bandwidthGBs = (memoryClockRateHz * busWidthBits / 8.0) / 1e9 * 2.0;

    printf("Device %d: %s\n", device, prop.name);
    printf("Memory Clock Rate (kHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Theoretical Peak Memory Bandwidth: %.3f GB/s\n", bandwidthGBs);

    return EXIT_SUCCESS;
}
```