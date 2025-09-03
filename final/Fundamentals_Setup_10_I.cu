```cpp
/*
Write a function `bool is_compute_capable(int major, int minor)` that queries the primary device and returns true if its compute capability is greater than or equal to the specified version.

Thinking process:
1. Include the necessary CUDA runtime header (`cuda_runtime.h`) to access device properties.
2. The function will use `cudaGetDeviceProperties` to obtain `cudaDeviceProp` for the current (primary) device, which is typically device 0.
3. Extract `major` and `minor` fields from the `cudaDeviceProp` structure.
4. Compare the device's compute capability with the requested major and minor values:
   - The device is capable if its major version is greater than the requested major.
   - If the major versions are equal, the device must have a minor version greater than or equal to the requested minor.
5. Return `true` if the condition is satisfied, otherwise `false`.
6. Add a simple `main` function to demonstrate usage and print the result for a test case.
*/

#include <cuda_runtime.h>
#include <iostream>

bool is_compute_capable(int major, int minor)
{
    cudaDeviceProp prop;
    // Get properties for the primary device (device 0)
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Check compute capability
    if (prop.major > major) {
        return true;
    }
    if (prop.major == major && prop.minor >= minor) {
        return true;
    }
    return false;
}

int main()
{
    // Example: check if device is at least compute capability 6.0
    int req_major = 6;
    int req_minor = 0;
    bool capable = is_compute_capable(req_major, req_minor);
    std::cout << "Device compute capability >= "
              << req_major << "." << req_minor << " : "
              << (capable ? "Yes" : "No") << std::endl;
    return 0;
}
```