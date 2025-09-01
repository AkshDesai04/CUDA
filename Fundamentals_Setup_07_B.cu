```cpp
/*
Given the maximum grid dimensions, calculate the theoretical maximum number of blocks you could launch in total (though this is often limited by other factors).

Thought Process:
1. The maximum grid dimensions are provided by the CUDA runtime via cudaDeviceProp.maxGridSize array.
2. We need to query the current device's properties using cudaGetDeviceProperties.
3. The theoretical maximum number of blocks is simply the product of the three dimensions:
      maxBlocks = maxGridSize[0] * maxGridSize[1] * maxGridSize[2].
   This works for all devices: older devices will have a third dimension of 1.
4. We will output both the grid dimensions and the computed maximum number of blocks.
5. Use long long for the product to avoid overflow on large dimensions.
6. Handle any CUDA errors gracefully.
7. Compile with nvcc as a .cu file.

*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device = 0;  // Use device 0; could be parameterized
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Retrieve maximum grid dimensions
    int maxX = prop.maxGridSize[0];
    int maxY = prop.maxGridSize[1];
    int maxZ = prop.maxGridSize[2];

    // Compute theoretical maximum number of blocks
    long long maxBlocks = static_cast<long long>(maxX) *
                          static_cast<long long>(maxY) *
                          static_cast<long long>(maxZ);

    // Output results
    std::cout << "Maximum grid dimensions: (" << maxX << ", " << maxY << ", " << maxZ << ")\n";
    std::cout << "Theoretical maximum number of blocks: " << maxBlocks << std::endl;

    return 0;
}
```