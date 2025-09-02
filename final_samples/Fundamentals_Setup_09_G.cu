```cuda
/*
 * Check if the device supports global L1 cache (`globalL1CacheSupported`).
 *
 * Thinking:
 * 1. The goal is to query the CUDA device properties and report whether the
 *    device supports a global L1 cache. The CUDA runtime provides a
 *    `cudaDeviceProp` structure that contains a member named
 *    `globalL1CacheSupported`.  According to the CUDA Toolkit documentation,
 *    this member is an `int` (or `bool` in newer headers) that is non‑zero if
 *    the device supports a global L1 cache.
 *
 * 2. We'll write a small program that:
 *    - Calls `cudaGetDeviceCount` to ensure at least one CUDA device is
 *      present.
 *    - For each device (or just device 0 for brevity), calls
 *      `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
 *    - Prints the device name, compute capability, and the value of
 *      `globalL1CacheSupported`.
 *
 * 3. Error handling: use a helper macro to check the return value of each
 *    CUDA API call and abort if an error occurs, printing the error string.
 *
 * 4. The program will be compiled with `nvcc` as a .cu file.  No external
 *    dependencies are required beyond the CUDA runtime.
 *
 * 5. Since the user requested only the .cu file, we provide the full code
 *    below, starting from the comment header and ending with `return 0;`.
 */

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__\
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Check for each device; for brevity, we report device 0 only.
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    std::cout << "Device " << dev << ": " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global L1 Cache Supported: "
              << (prop.globalL1CacheSupported ? "Yes" : "No") << std::endl;

    return 0;
}
```