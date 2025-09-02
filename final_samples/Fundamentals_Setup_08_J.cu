/*
Use `cudaGetDeviceProperties` and print the `driverVersion` field from the properties struct.

Thinking:
- The goal is to query the CUDA runtime for device properties and specifically output the `driverVersion` field.
- We will include the necessary CUDA header (`cuda_runtime.h`) and iostream for printing.
- We'll write a small helper macro to check CUDA errors for clean error handling.
- In `main`, we will get the number of devices with `cudaGetDeviceCount`, iterate over each device, obtain its properties via `cudaGetDeviceProperties`, and print the `driverVersion`.
- The `driverVersion` is the same for all devices on a system, but printing it for each device confirms the query works.
- Finally, we will return 0 after printing.
*/

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return EXIT_FAILURE;                                       \
        }                                                              \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << " (" << prop.name << "): "
                  << "driverVersion = " << prop.driverVersion << std::endl;
    }

    return EXIT_SUCCESS;
}
