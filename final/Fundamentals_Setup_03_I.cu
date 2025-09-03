```cpp
/*
 * Calculate and print the theoretical peak memory bandwidth of the primary device in GB/s.
 * Formula: (memoryClockRate_kHz * 1000 * memoryBusWidth_bits / 8) / 1e9 * 2
 * (the *2 is for DDR memory).
 *
 * Thinking:
 * - Use the CUDA Runtime API to query device properties via cudaGetDeviceProperties.
 * - The primary device is typically device 0. We can also check the number of devices.
 * - The memoryClockRate field is reported in kHz, so we multiply by 1000 to convert to Hz.
 * - memoryBusWidth is in bits; dividing by 8 converts it to bytes.
 * - The factor of 2 accounts for DDR memory (double data rate).
 * - Convert the result to GB/s by dividing by 1e9 (bytes per second to gigabytes per second).
 * - Handle errors from CUDA API calls and output a clear message if no device is found.
 * - Print the result with reasonable precision (e.g., 2 decimal places).
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to get device count. %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    // Choose the primary device (device 0)
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to get device properties. %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Compute theoretical peak memory bandwidth in GB/s
    double memoryClockRateHz = static_cast<double>(prop.memoryClockRate) * 1000.0; // Convert kHz to Hz
    double busWidthBytes = static_cast<double>(prop.memoryBusWidth) / 8.0;         // Convert bits to bytes
    double bandwidthGBs = (memoryClockRateHz * busWidthBytes) / 1e9 * 2.0;       // DDR factor

    printf("Device %d (%s)\n", device, prop.name);
    printf("Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Theoretical Peak Memory Bandwidth: %.2f GB/s\n", bandwidthGBs);

    return EXIT_SUCCESS;
}
```