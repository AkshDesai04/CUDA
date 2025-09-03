```cpp
/*
Aim of the program: For each device, print its memory bus width in bits.

Thinking process:
1. The requirement is straightforward: enumerate all CUDA-capable devices available on the system,
   retrieve each device's properties, and output the memory bus width in bits.
2. The CUDA Runtime API provides the function `cudaGetDeviceCount` to obtain the number of devices
   and `cudaGetDeviceProperties` to obtain a `cudaDeviceProp` structure containing various
   hardware characteristics, including the field `memoryBusWidth` which is expressed in bits.
3. We'll write a simple main function that:
   - Calls `cudaGetDeviceCount`.
   - Loops over each device index.
   - Calls `cudaGetDeviceProperties` for that index.
   - Prints the device index, name, and the memory bus width.
4. Basic error checking will be included: if `cudaGetDeviceCount` or `cudaGetDeviceProperties`
   fails, the program will report the error and exit.
5. Since the user asked for a .cu file, we write the code with .cu extension in mind,
   include <cuda_runtime.h>, and use standard C++ i/o.
6. The final program is self-contained and can be compiled with `nvcc` and executed
   on a system with CUDA installed.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << dev << ": "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
    }

    return EXIT_SUCCESS;
}
```