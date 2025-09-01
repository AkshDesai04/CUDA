```c
/*
Aim: Check if the device supports local L1 cache (`localL1CacheSupported`).

Thinking:
The CUDA runtime API provides several ways to query device properties. One of
the attributes that indicates whether the device supports a local L1 cache is
`CUDA_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED`.  This attribute is
accessible via `cudaDeviceGetAttribute`.  The value returned is an integer
(0 or 1).  If the attribute is not available on a particular compute
capability, the API will return an error code which we can handle.

The program will:
1. Query the number of CUDA devices available on the system.
2. Iterate over each device, retrieving the `localL1CacheSupported` attribute.
3. Print a human‑readable message indicating whether each device supports
   a local L1 cache.
4. Use proper error handling to report any CUDA runtime errors.

A small helper function `checkCudaError` will simplify error checking
by printing the error string and aborting the program if a CUDA call fails.

The code is written in plain CUDA C, suitable for compilation with
`nvcc`.  It includes the standard headers and uses `std::cout` for output.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " Error: " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Failed to get device count");

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        int localL1Supported = 0;
        cudaError_t err = cudaDeviceGetAttribute(&localL1Supported,
            cudaDevAttrLocalL1CacheSupported, dev);
        if (err == cudaSuccess) {
            std::cout << "Device " << dev << " local L1 cache supported: "
                      << (localL1Supported ? "Yes" : "No") << std::endl;
        } else {
            // If the attribute is not supported for this device, report it.
            std::cout << "Device " << dev << " local L1 cache supported: "
                      << "Attribute not supported (error: "
                      << cudaGetErrorString(err) << ")" << std::endl;
        }
    }

    return 0;
}
```