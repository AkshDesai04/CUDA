/*
Aim of the program: Get the name of the current device and the length of the name string.

Thinking:
The program is written in C++ for CUDA. We need to query the currently active CUDA device, retrieve its properties, and then print out two pieces of information: the device name and the length of that name string.

Key steps:
1. Use `cudaGetDevice()` to obtain the ID of the currently active device.
2. Use `cudaGetDeviceProperties()` to get a `cudaDeviceProp` structure which contains a `name` field (a null-terminated string) and the device's capabilities.
3. The `name` string is 256 characters long in the struct. The actual string is null-terminated, so we can use `strlen()` to compute its length.
4. Output the device name and the computed length using standard C++ I/O (`std::cout`).
5. Check CUDA API return values and handle errors gracefully.

We include `<cuda_runtime.h>` for CUDA API functions, `<iostream>` for output, and `<cstring>` for `strlen`. A simple macro `CUDA_CHECK` is defined to simplify error handling.

The program runs on the host only (no device kernels). It will compile with `nvcc` and run on any machine with CUDA installed.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

#define CUDA_CHECK(err)                                                \
    do {                                                               \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                      << " at line " << __LINE__ << std::endl;         \
            exit(EXIT_FAILURE);                                       \
        }                                                              \
    } while (0)

int main()
{
    int deviceId = 0;
    cudaError_t err = cudaGetDevice(&deviceId);
    CUDA_CHECK(err);

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, deviceId);
    CUDA_CHECK(err);

    const char* deviceName = prop.name;
    size_t nameLength = std::strlen(deviceName);

    std::cout << "Device name: " << deviceName << std::endl;
    std::cout << "Name length: " << nameLength << std::endl;

    return 0;
}
