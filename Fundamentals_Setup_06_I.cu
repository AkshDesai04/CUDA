```cpp
/* 
Aim of the program: 
Query and print the maximum layered 1D texture size and number of layers. 

Thinking Process:
1. CUDA exposes texture-related limits through the cudaDeviceProp structure.
2. For 1D layered textures, two relevant fields exist:
   - `maxTexture1DLayered`: The maximum width (size) of a 1D layered texture.
   - `maxTexture1DMipmappableLayers`: The maximum number of layers in a 1D mipmapped layered texture.
     For non-mipmapped layered textures, the layer count is typically bounded by the same limit,
     but the dedicated field provides the explicit number of layers.
3. The program will:
   - Retrieve the number of CUDA-capable devices.
   - For each device, obtain its properties with `cudaGetDeviceProperties`.
   - Print the device name, the maximum 1D layered texture width, and the maximum number of layers.
4. Simple error checking will be performed after each CUDA call.
5. The program is self‑contained and can be compiled with `nvcc` to produce a `.cu` file.

*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Maximum 1D layered texture width (elements): %d\n", prop.maxTexture1DLayered);
        printf("  Maximum number of layers for 1D layered textures: %d\n", prop.maxTexture1DMipmappableLayers);
        printf("\n");
    }

    return EXIT_SUCCESS;
}
```