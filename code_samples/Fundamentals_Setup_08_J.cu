```cpp
/* 
Use `cudaGetDeviceProperties` and print the `driverVersion` field from the properties struct. 

I am going to write a minimal CUDA C++ program that queries the properties of a CUDA device using cudaGetDeviceProperties, then prints the driverVersion field from the returned cudaDeviceProp structure. The aim is to demonstrate how to access device properties and retrieve specific information such as the driver version. I will include necessary headers, use the CUDA Runtime API, and perform basic error checking. The program will work with a single device (device 0) and will print the driver version to standard output. 

The steps are:
1. Include <cuda_runtime.h> and <iostream> for CUDA and I/O.
2. Define a helper macro to check CUDA errors and abort on failure.
3. In main(), get the number of devices. If none, exit with a message.
4. Call cudaGetDeviceProperties on device 0 to obtain a cudaDeviceProp structure.
5. Extract and print the driverVersion field.
6. Return 0.

This satisfies the requirement and is a complete, compilable .cu file. 
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(err) << std::endl;   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                  \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));

    std::cout << "Driver version for device 0: " << deviceProp.driverVersion << std::endl;

    return EXIT_SUCCESS;
}
```