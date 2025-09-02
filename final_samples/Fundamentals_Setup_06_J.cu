```cpp
/*
Aim: Query and print the maximum layered 2D texture size and number of layers.

Thinking:
To obtain the maximum layered 2D texture size and the maximum number of layers supported by a CUDA-capable device, we need to query the device properties via the CUDA Runtime API. The struct `cudaDeviceProp` contains fields `maxTexture2DLayered` and `maxTextureArrayLayers` which represent these limits. The program will:
1. Use `cudaGetDeviceCount` to determine how many CUDA devices are available.
2. For each device, call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` structure.
3. Print out the device name, the maximum 2D layered texture size (interpreted as both width and height), and the maximum number of layers that can be in a 2D layered texture (the `maxTextureArrayLayers` field).
4. Handle errors gracefully using a simple macro.
The output will be human‑readable console text. This program compiles as a standard CUDA C++ source file (`.cu`), requiring only `nvcc` to build.

Note: The `maxTexture2DLayered` field gives the maximum dimension (width/height) for a 2D layered texture; it is the same for both axes. The `maxTextureArrayLayers` field gives the maximum number of layers that can be stored in a 2D array (i.e., layered texture).
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA API call results
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s (%d)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);          \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("\nDevice %d: %s\n", dev, prop.name);

        // Maximum 2D layered texture size (width and height)
        printf("  Maximum 2D layered texture size: %d (width x height)\n",
               prop.maxTexture2DLayered);

        // Maximum number of layers for a 2D layered texture (texture array)
        printf("  Maximum layers per texture array: %d\n",
               prop.maxTextureArrayLayers);
    }

    return EXIT_SUCCESS;
}
```