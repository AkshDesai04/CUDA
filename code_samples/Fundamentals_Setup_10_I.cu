/*
Aim: Write a function `bool is_compute_capable(int major, int minor)` that queries the primary device and returns true if its compute capability is greater than or equal to the specified version.

Thinking:
- We need to determine the compute capability (major, minor) of the currently active CUDA device.
- The primary device can be retrieved using `cudaGetDevice`, which returns the current device ID.
- `cudaGetDeviceProperties` fills a `cudaDeviceProp` structure that contains the device's major and minor compute capability values.
- After retrieving the properties, compare the device's major/minor with the supplied thresholds:
    * If the device's major is greater than the requested major, the device is certainly compute capable.
    * If the majors are equal, we compare the minor numbers.
- Error handling:
    * If either `cudaGetDevice` or `cudaGetDeviceProperties` fails, we conservatively return `false`.
- The function returns a C `bool` (from `<stdbool.h>`), not a CUDA `bool` type.
- No kernel launches or device code are needed; everything is host-side.
- The code is written as a standalone .cu file that can be compiled with nvcc.
*/

#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>

/**
 * @brief Checks if the primary CUDA device has a compute capability
 *        greater than or equal to the specified major/minor version.
 *
 * @param major Requested major compute capability.
 * @param minor Requested minor compute capability.
 * @return true  if the device's compute capability is >= requested version.
 * @return false otherwise or if an error occurs while querying the device.
 */
bool is_compute_capable(int major, int minor) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        // Could not get current device
        return false;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        // Could not retrieve device properties
        return false;
    }

    if (prop.major > major) {
        return true;
    } else if (prop.major == major && prop.minor >= minor) {
        return true;
    } else {
        return false;
    }
}

/* Optional test main function */
/*
int main() {
    int req_major = 6, req_minor = 0;
    bool capable = is_compute_capable(req_major, req_minor);
    printf("Device compute capability >= %d.%d : %s\n",
           req_major, req_minor, capable ? "Yes" : "No");
    return 0;
}
*/

