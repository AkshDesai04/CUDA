/*
Get the name of the current device and the length of the name string.

Thinking:
- The CUDA runtime API allows us to query properties of a device via cudaGetDeviceProperties.
- The cudaDeviceProp struct contains a 'name' field which is a fixed-size char array storing the device name.
- The current active device can be obtained with cudaGetDevice, but typically the default device 0 is used.
- We can then compute the string length using strlen from <cstring>.
- Finally, print the name and its length.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

int main() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get current device: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    const char* name = prop.name;
    size_t len = std::strlen(name);

    std::cout << "Device Name: " << name << std::endl;
    std::cout << "Name Length: " << len << std::endl;

    return 0;
}
