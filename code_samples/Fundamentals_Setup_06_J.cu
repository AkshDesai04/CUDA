```cuda
/*
Aim of the program:
Query and print the maximum layered 2D texture size and number of layers.

Thought process and design decisions:
1. The CUDA Runtime API provides device properties in a `cudaDeviceProp` struct. 
   It contains fields `maxTexture2DLayeredWidth`, `maxTexture2DLayeredHeight`, 
   and `maxTexture2DLayeredLayers` that describe the limits for layered 2D textures. 
   These are the exact values we need to retrieve.

2. The program will:
   - Retrieve the number of CUDA-capable devices using `cudaGetDeviceCount`.
   - For each device:
     - Get its properties via `cudaGetDeviceProperties`.
     - Extract the three relevant fields.
     - Print them in a human‑readable format, along with the device name for clarity.
   - Handle any CUDA errors gracefully by checking the return status of each CUDA API call.

3. Since we only need to query device properties (no kernel launch), this can be a plain
   console application. The file will be a standard `.cu` file and can be compiled with
   `nvcc`.

4. The code uses `iostream` for output. All CUDA API calls are wrapped with a simple
   macro `CHECK_CUDA` that prints the error and exits if an error occurs.

5. Edge cases: Some older GPUs may not support layered textures, in which case the
   maximum layered width, height, and layers may be zero. The program will still print
   the values, letting the user see if layered textures are supported.

6. The program is self‑contained and contains no external dependencies beyond the CUDA
   Runtime library and the standard C++ library.
*/

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro to check CUDA errors
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":"            \
                      << __LINE__ << " - " << cudaGetErrorString(err)   \
                      << std::endl;                                    \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Max Layered 2D Width  : " << prop.maxTexture2DLayeredWidth << std::endl;
        std::cout << "  Max Layered 2D Height : " << prop.maxTexture2DLayeredHeight << std::endl;
        std::cout << "  Max Layered 2D Layers : " << prop.maxTexture2DLayeredLayers << std::endl;
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
```