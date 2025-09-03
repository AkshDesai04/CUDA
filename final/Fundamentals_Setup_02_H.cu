```cuda
/*
 * Aim: For each device, print its L2 cache size in megabytes (MB).
 *
 * Thinking:
 * 1. We need to query all CUDA-capable devices present in the system.
 * 2. Use `cudaGetDeviceCount` to obtain the number of devices.
 * 3. For each device index, call `cudaGetDeviceProperties` to retrieve a
 *    `cudaDeviceProp` structure. This structure contains a field
 *    `l2CacheSize` which represents the L2 cache size in bytes.
 * 4. Convert the size from bytes to megabytes by dividing by (1024 * 1024).
 * 5. Output the device index and the computed size.  Use `std::cout`
 *    for human‑readable output.
 * 6. Include proper error checking for CUDA API calls.  If a call
 *    fails, print an error message and exit.
 * 7. Since the task is simple, we implement everything in `main`.
 * 8. The program will be compiled with `nvcc` and produce an executable.
 *
 * Note: The `l2CacheSize` field is available on devices that expose an
 * L2 cache.  On older or very small devices it may be zero; we still
 * print the value for completeness.
 */

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Error: cudaGetDeviceProperties for device "
                      << dev << " failed: " << cudaGetErrorString(err)
                      << std::endl;
            continue;
        }

        // l2CacheSize is in bytes; convert to MB
        double l2MB = static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0);

        std::cout << "Device " << dev << ": L2 cache size = "
                  << l2MB << " MB" << std::endl;
    }

    return 0;
}
```