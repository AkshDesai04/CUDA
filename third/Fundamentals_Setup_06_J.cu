/*
Query and print the maximum layered 2D texture size and number of layers.

Thinking:
- The CUDA runtime provides a struct cudaDeviceProp that contains the max size of layered 2D textures via the array maxTexture2DLayered[3], where indices 0 and 1 represent the width and height, and index 2 represents the maximum number of layers.
- We'll query the device properties using cudaGetDeviceProperties.
- We will iterate over all devices (if more than one) and print out the relevant values.
- Include error checking for CUDA API calls.
- Output will be simple text printed to std::cout.
*/

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    if (deviceCount == 0)
    {
        std::cout << "No CUDA-capable device detected." << std::endl;
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            std::cerr << "Error getting properties for device " << dev << ": "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Max layered 2D texture width  : " << prop.maxTexture2DLayered[0] << std::endl;
        std::cout << "  Max layered 2D texture height : " << prop.maxTexture2DLayered[1] << std::endl;
        std::cout << "  Max number of layers          : " << prop.maxTexture2DLayered[2] << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
