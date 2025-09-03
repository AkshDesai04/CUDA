```cpp
/*
 * Aim of the program (as requested, word for word):
 * "Given the maximum grid dimensions, calculate the theoretical maximum number of blocks you could launch in total (though this is often limited by other factors)."
 *
 * Thinking process:
 * 1. We need to query the device properties to obtain the maximum grid dimensions. In CUDA, this is accessible via the cudaDeviceProp struct, specifically the maxGridSize array which contains the limits for x, y, and z dimensions.
 * 2. The theoretical maximum number of blocks that can be launched is simply the product of these three dimensions. Some GPUs may only support 2D grid (maxGridSize[2] == 0), but multiplying by 0 would give 0, so we treat such cases by ignoring the zero dimension: if maxGridSize[2] == 0, we use 1 for that dimension to reflect a 2D grid.
 * 3. We'll write a small utility to check CUDA errors.
 * 4. The program will print the maximum grid size per dimension, then print the computed theoretical maximum number of blocks.
 * 5. This is a straightforward C++ CUDA program that can be compiled with nvcc.
 */

#include <iostream>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main() {
    // Get number of CUDA-capable devices
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found." << std::endl;
        return EXIT_FAILURE;
    }

    // Use device 0 for this query
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxGridX = prop.maxGridSize[0];
    int maxGridY = prop.maxGridSize[1];
    int maxGridZ = prop.maxGridSize[2];

    std::cout << "Maximum grid dimensions (x, y, z): "
              << maxGridX << ", " << maxGridY << ", " << maxGridZ << std::endl;

    // For 2D grid support, treat z dimension as 1 if it's 0
    int zDim = (maxGridZ == 0) ? 1 : maxGridZ;

    // Calculate theoretical maximum number of blocks
    long long theoreticalMaxBlocks = static_cast<long long>(maxGridX) *
                                     static_cast<long long>(maxGridY) *
                                     static_cast<long long>(zDim);

    std::cout << "Theoretical maximum number of blocks that can be launched: "
              << theoreticalMaxBlocks << std::endl;

    return EXIT_SUCCESS;
}
```