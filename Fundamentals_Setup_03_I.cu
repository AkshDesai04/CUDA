```cpp
/*
Aim: Calculate and print the theoretical peak memory bandwidth of the primary device in GB/s. Formula: (memoryClockRate_kHz * 1000 * memoryBusWidth_bits / 8) / 1e9 * 2 (the *2 is for DDR memory).

Thinking:
1. We need to query the primary CUDA device (device 0) for its memory clock rate (in kHz) and memory bus width (in bits). CUDA provides these in the `cudaDeviceProp` structure, accessible via `cudaGetDeviceProperties`.
2. The formula for peak memory bandwidth is derived from the standard memory transfer formula:
   Bandwidth (bytes/s) = (Clock rate in Hz * Bus width in bits / 8) * 2
   The factor 2 accounts for DDR (double data rate) memory.
   We convert the memory clock rate from kHz to Hz by multiplying by 1,000.
3. Once we compute the bandwidth in bytes per second, we divide by 1e9 to convert to gigabytes per second (GB/s).
4. The program will:
   - Initialize CUDA, get device properties.
   - Compute the bandwidth.
   - Print the result with a reasonable number of decimal places.
5. We'll handle any potential CUDA errors and print an error message if the device query fails.
6. The program will be compiled as a CUDA C++ source file (`.cu`), but the logic itself is purely CPU side, no kernel launches needed.
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Get the number of CUDA devices
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    // Use the primary device (device 0)
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Error getting device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // memoryClockRate is in kHz, memoryBusWidth is in bits
    double memoryClockRate_kHz = static_cast<double>(prop.memoryClockRate);
    double memoryBusWidth_bits = static_cast<double>(prop.memoryBusWidth);

    // Compute peak memory bandwidth in GB/s
    double bandwidth_GB_per_s = (memoryClockRate_kHz * 1000.0 * memoryBusWidth_bits / 8.0) / 1e9 * 2.0;

    std::cout << "Theoretical peak memory bandwidth of device " << device
              << " is: " << bandwidth_GB_per_s << " GB/s" << std::endl;

    return 0;
}
```