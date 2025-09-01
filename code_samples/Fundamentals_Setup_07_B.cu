```c
/* 
Aim: 
Given the maximum grid dimensions, calculate the theoretical maximum number of blocks you could launch in total (though this is often limited by other factors).

Thinking:
The CUDA runtime API exposes the maximum supported grid dimensions for a device via the `cudaDeviceProp.maxGridSize` array. This array contains three integers, representing the maximum grid size in the x, y, and z dimensions. The theoretical maximum number of blocks that can be launched in a single kernel invocation is simply the product of these three dimensions: 
    maxBlocks = maxGridSize[0] * maxGridSize[1] * maxGridSize[2].
For devices that only support 2D grids, the third dimension will be 1, which naturally falls into the product calculation.

To obtain these values, we:
1. Query the number of CUDA-capable devices.
2. Choose the first device (device 0) for simplicity.
3. Retrieve the device properties via `cudaGetDeviceProperties`.
4. Extract the `maxGridSize` array.
5. Compute the product as an unsigned long long to avoid overflow on very large GPUs.
6. Print the result.

The program is intentionally simple: it does not launch any kernel; it only queries device properties and prints the theoretical maximum block count. This makes the program portable across all CUDA-capable GPUs and allows quick assessment of the theoretical launch limits.

Implementation details:
- Use `<cuda_runtime.h>` for CUDA API calls.
- Use `<iostream>` for output.
- Check return codes of CUDA API functions and exit on error.
- Provide meaningful error messages if no CUDA devices are found.

This approach yields the required value in a self-contained CUDA C program compiled with nvcc. 
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable devices found." << std::endl;
        return 1;
    }

    // For simplicity, use device 0
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Retrieve maximum grid dimensions
    int maxGridX = prop.maxGridSize[0];
    int maxGridY = prop.maxGridSize[1];
    int maxGridZ = prop.maxGridSize[2];

    unsigned long long maxBlocks = static_cast<unsigned long long>(maxGridX) *
                                   static_cast<unsigned long long>(maxGridY) *
                                   static_cast<unsigned long long>(maxGridZ);

    std::cout << "Device " << device << " (" << prop.name << "):" << std::endl;
    std::cout << "  Max grid size (x, y, z): (" << maxGridX << ", " << maxGridY << ", " << maxGridZ << ")" << std::endl;
    std::cout << "  Theoretical maximum number of blocks per launch: " << maxBlocks << std::endl;

    return 0;
}
```