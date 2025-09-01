```cuda
/*
Aim: Query and print the maximum texture dimension for a
1D texture using cudaDeviceGetAttribute with cudaDevAttrMaxTexture1DWidth.

Thinking:
The program is a simple CUDA host code that uses the CUDA runtime API to
retrieve the maximum supported width for 1D textures on each available GPU.
The relevant API is cudaDeviceGetAttribute which takes an attribute enum,
a device ID, and returns the attribute value. The attribute enum
cudaDevAttrMaxTexture1DWidth specifies the maximum width of a 1D texture.

Steps:
1. Include necessary headers: <cuda_runtime.h> for CUDA runtime functions,
   <iostream> for printing.
2. In main():
   a. Query the number of CUDA devices with cudaGetDeviceCount.
   b. If no devices are found, print a message and exit.
   c. Loop over each device ID from 0 to deviceCount-1.
   d. For each device, call cudaDeviceGetAttribute with
      cudaDevAttrMaxTexture1DWidth.
   e. Handle any errors from cudaDeviceGetAttribute and print a
      descriptive error message.
   f. If successful, print the device number and the maximum texture
      1D width.
3. Use proper error checking for each CUDA call. The helper macro
   CUDA_CHECK will wrap calls and abort on failure with a message.
4. Compile with nvcc and run. It will print the maximum 1D texture width
   for each GPU in the system.

Edge cases:
- Devices that do not support 1D textures? In practice all GPUs support
  1D textures; the attribute should be valid. If an attribute is not
  supported, CUDA will return cudaErrorInvalidValue, which we handle.
- Very large values are represented as int; on recent GPUs the value
  can be >2^31, so we cast to size_t for printing to avoid overflow.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << " : " << cudaGetErrorString(err) << std::endl;          \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        size_t maxTexture1DWidth = 0;
        cudaError_t err = cudaDeviceGetAttribute(
            reinterpret_cast<int*>(&maxTexture1DWidth),
            cudaDevAttrMaxTexture1DWidth,
            dev);

        if (err != cudaSuccess) {
            std::cerr << "Failed to get cudaDevAttrMaxTexture1DWidth for device "
                      << dev << " : " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev
                  << " : Maximum 1D texture width = "
                  << maxTexture1DWidth << std::endl;
    }

    return EXIT_SUCCESS;
}
```