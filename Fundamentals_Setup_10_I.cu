```cpp
/*
 * Write a function `bool is_compute_capable(int major, int minor)` that queries the primary device and returns true if its compute capability is greater than or equal to the specified version.
 *
 * Thinking:
 * 1. We need to query the current device's properties using CUDA runtime API.
 * 2. The "primary device" refers to the device currently set with cudaSetDevice() or the default device (0 if none set).
 * 3. Use cudaGetDevice() to obtain the device id, then cudaGetDeviceProperties() to retrieve its major and minor compute capability.
 * 4. Compare the retrieved major/minor with the required major/minor: 
 *    - If retrieved major > required major → true.
 *    - If major equal and retrieved minor >= required minor → true.
 *    - Otherwise → false.
 * 5. Handle any CUDA errors gracefully: if retrieving properties fails, return false.
 * 6. The function will be pure C++ (CUDA runtime) and can be called from any host code.
 */

#include <cuda_runtime.h>
#include <stdbool.h>

bool is_compute_capable(int major, int minor) {
    // Get current device
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        // If we cannot get the device, assume not compute capable
        return false;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        // If we cannot get properties, assume not compute capable
        return false;
    }

    // Compare compute capability
    if (prop.major > major) return true;
    if (prop.major == major && prop.minor >= minor) return true;
    return false;
}
```