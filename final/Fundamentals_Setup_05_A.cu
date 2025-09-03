/*
Aim of the program:
Based on the `maxThreadsPerBlock` value and a warp size of 32, calculate and print the maximum number of warps per block.

Thinking:
1. The maximum number of warps per block is simply the integer division of the device's maxThreadsPerBlock by the warp size (32). 
2. We obtain maxThreadsPerBlock via cudaGetDeviceProperties, which fills a cudaDeviceProp struct. 
3. The warp size is a constant in CUDA (32), so we can use that directly. 
4. We handle potential errors from cudaGetDeviceProperties by checking the return code. 
5. Finally, we print the computed number of warps using std::cout.
6. The program is minimal, includes necessary headers (<iostream>, <cuda_runtime.h>), and uses a single main function. 
7. It runs on the current device (device 0) unless changed; no additional device selection is required for this simple calculation. 
8. Because the division is integer, any remainder is truncated, which is fine because a warp must be full-size; in practice, maxThreadsPerBlock is a multiple of 32 on NVIDIA GPUs. 
9. The output format is straightforward: "Maximum number of warps per block: X".
*/

#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Get properties of the current device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "Error retrieving device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Warp size is 32
    const int warpSize = 32;
    // Compute maximum number of warps per block
    int maxWarpsPerBlock = prop.maxThreadsPerBlock / warpSize;

    std::cout << "Maximum number of warps per block: " << maxWarpsPerBlock << std::endl;
    return 0;
}
