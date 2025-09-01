/*
Write a function `bool is_compute_capable(int major, int minor)` that queries the primary device and returns true if its compute capability is greater than or equal to the specified version.

Thinking:
- Use CUDA runtime API to obtain current device ID with cudaGetDevice.
- Retrieve device properties using cudaGetDeviceProperties.
- Compare the returned major and minor compute capability values to the arguments.
- Return true when the device's major is greater, or if majors are equal and the device's minor is greater or equal.
- If any CUDA call fails, gracefully return false.
- Mark the function as __host__ to ensure it runs on the CPU.
- Provide a simple main for demonstration that prints whether the device is capable of a given compute capability.
*/

#include <cuda_runtime.h>
#include <cstdio>

__host__ bool is_compute_capable(int major, int minor) {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        return false;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        return false;
    }

    if (prop.major > major) return true;
    if (prop.major == major && prop.minor >= minor) return true;
    return false;
}

int main() {
    int req_major = 7;
    int req_minor = 0;
    bool capable = is_compute_capable(req_major, req_minor);
    printf("Device is %scompute capable of %d.%d\n", capable ? "" : "not ", req_major, req_minor);
    return 0;
}
