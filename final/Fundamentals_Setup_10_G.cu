```cpp
/*
Aim of the program: Query and print the `regsPerBlock` property.

Thought process:
- The `regsPerBlock` property is part of `cudaDeviceProp`, which can be obtained via
  `cudaGetDeviceProperties()`. It indicates the number of registers per block that
  the device can allocate.
- We will write a CUDA C++ program that queries the number of devices using
  `cudaGetDeviceCount()`. For each device, we call `cudaGetDeviceProperties()` to
  retrieve the device properties struct.
- Then we print the `regsPerBlock` value for each device to the console.
- Error checking will be performed after each CUDA API call. If an error occurs,
  we will print an error message and exit the program.
- The program uses standard C++ I/O (`iostream`) to display results.

The code below is self-contained and should compile with `nvcc`. It requires a CUDA
enabled compiler and a compatible device.

*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Error: cudaGetDeviceProperties failed for device " << dev
                      << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << " (" << prop.name << "):" << std::endl;
        std::cout << "  regsPerBlock = " << prop.regsPerBlock << std::endl;
    }

    return EXIT_SUCCESS;
}
```