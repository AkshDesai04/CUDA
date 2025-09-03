/*
 * Query and print the maximum layered 2D texture size and number of layers.
 *
 * --------------------------------------------------------------------
 * Thought process:
 * 1. The goal is to obtain the device capabilities related to layered 2D textures.
 *    CUDA exposes this information in the `cudaDeviceProp` structure, specifically
 *    in the `maxTexture2DLayered` array (width, height, layers).
 * 2. Steps:
 *    a. Retrieve the number of CUDA devices (`cudaGetDeviceCount`).
 *    b. Select a device (here we use device 0 for simplicity).
 *    c. Query the device properties with `cudaGetDeviceProperties`.
 *    d. Extract `maxTexture2DLayered` values.
 *    e. Print them to the console.
 * 3. Error handling:
 *    Wrap CUDA API calls in a macro that checks the returned `cudaError_t`
 *    and aborts with a meaningful message if an error occurs.
 * 4. Use C++ i/o (`std::cout`) for nicer formatting, but the program
 *    can be compiled with `nvcc` just like any other CUDA source file.
 * 5. The program is small and self‑contained; no additional files or libraries
 *    are required.
 *
 * Compile with:
 *    nvcc -o query_layered_texture query_layered_texture.cu
 *
 * Run:
 *    ./query_layered_texture
 *
 * The output will display the maximum width, height, and number of layers
 * supported for 2D layered textures on the selected CUDA device.
 * --------------------------------------------------------------------
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA error in " << __FILE__ << ":"          \
                      << __LINE__ << " - " << cudaGetErrorString(err)\
                      << std::endl;                                  \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    int device = 0; // Use the first device
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
    std::cout << "Maximum layered 2D texture size:" << std::endl;
    std::cout << "  Width   : " << deviceProp.maxTexture2DLayered[0] << std::endl;
    std::cout << "  Height  : " << deviceProp.maxTexture2DLayered[1] << std::endl;
    std::cout << "  Layers  : " << deviceProp.maxTexture2DLayered[2] << std::endl;

    return EXIT_SUCCESS;
}
