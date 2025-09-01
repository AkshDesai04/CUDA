```cpp
/* 
For each device, print its maximum grid dimensions ([X, Y, Z]).

Thought process:
1. We need to query the number of CUDA-capable devices on the system.
2. For each device, obtain its device properties via `cudaGetDeviceProperties`.
3. The `cudaDeviceProp` structure contains `maxGridSize` array with 3 elements for X, Y, Z dimensions.
4. Print these dimensions in a readable format.
5. Handle errors gracefully: check the return codes of CUDA API calls and report any failures.
6. Use standard C++ I/O for simplicity, since the task is purely console output.
7. Wrap the code in a `main` function and compile with nvcc as a `.cu` file.

This program demonstrates basic CUDA device introspection and prints the maximum grid dimensions for each GPU.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get properties for device " << dev << ": "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << " (" << prop.name << "): "
                  << "Max Grid Dimensions = ["
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << "]" << std::endl;
    }

    return 0;
}
```