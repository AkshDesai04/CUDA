```cuda
/*
Aim of the program:
From the maximum dimensions [X, Y, Z], verify that X * Y * Z is not necessarily equal to `maxThreadsPerBlock` (since X, Y, and Z are individual dimension limits).

Thinking:
- The CUDA driver exposes the maximum number of threads per block as `maxThreadsPerBlock` in the `cudaDeviceProp` structure.
- It also provides the maximum dimensional limits for a block: `maxThreadsDim.x`, `maxThreadsDim.y`, and `maxThreadsDim.z`.
- These dimension limits are separate; each dimension can be at its own maximum, but the product of these three maximums can exceed `maxThreadsPerBlock`.
- The goal of the program is to query a device's properties, compute the product `maxThreadsDim.x * maxThreadsDim.y * maxThreadsDim.z`, and compare it to `maxThreadsPerBlock`.
- We'll print all relevant values and indicate whether the equality holds. This demonstrates that the product of individual dimension limits is not a guarantee for the overall maximum thread count per block.
*/

#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    // For simplicity, use the first device
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "  maxThreadsPerBlock   : " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  maxThreadsDim.x      : " << prop.maxThreadsDim.x << std::endl;
    std::cout << "  maxThreadsDim.y      : " << prop.maxThreadsDim.y << std::endl;
    std::cout << "  maxThreadsDim.z      : " << prop.maxThreadsDim.z << std::endl;

    size_t dimProduct = static_cast<size_t>(prop.maxThreadsDim.x) *
                        static_cast<size_t>(prop.maxThreadsDim.y) *
                        static_cast<size_t>(prop.maxThreadsDim.z);

    std::cout << "  Product of maxThreadsDim (x*y*z) : " << dimProduct << std::endl;

    if (dimProduct == static_cast<size_t>(prop.maxThreadsPerBlock)) {
        std::cout << "Result: The product of individual dimension limits equals maxThreadsPerBlock." << std::endl;
    } else {
        std::cout << "Result: The product of individual dimension limits does NOT equal maxThreadsPerBlock." << std::endl;
        std::cout << "  Explanation: The limits are independent per dimension; the product can exceed the overall per-block thread limit." << std::endl;
    }

    return EXIT_SUCCESS;
}
```