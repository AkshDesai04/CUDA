/*
Aim of the program: Query and print whether the device supports stream priorities.

Thinking:
The CUDA runtime API exposes a device property called `streamPrioritiesSupported` that indicates
whether a device supports stream priorities.  We can retrieve this information by calling
`cudaGetDeviceProperties()` and inspecting the `streamPrioritiesSupported` member of
`cudaDeviceProp`.  An alternative is to use `cudaDeviceGetAttribute()` with the
`cudaDevAttrStreamPrioritiesSupported` attribute, but the property approach is clearer
and works across all CUDA versions that support the feature.

The program will:
1. Query the number of CUDA-capable devices.
2. For each device, retrieve its properties.
3. Print whether stream priorities are supported for that device.
4. Handle any CUDA errors with a helper macro `CUDA_CHECK`.

This keeps the implementation simple and demonstrates the feature query cleanly.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t e = (err);                                               \
        if (e != cudaSuccess) {                                             \
            std::cerr << "CUDA error (" << e << "): "                       \
                      << cudaGetErrorString(e) << " at " << __FILE__        \
                      << ":" << __LINE__ << std::endl;                      \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Stream priorities supported: "
                  << (prop.streamPrioritiesSupported ? "Yes" : "No") << std::endl;
    }

    return 0;
}
