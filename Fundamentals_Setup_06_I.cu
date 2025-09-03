```cpp
/*
Aim of the program: Query and print the maximum layered 1D texture size and number of layers.

Thinking:
The CUDA runtime exposes texture limits through the cudaDeviceProp structure. 
Two relevant fields are:
- maxTexture1DLayered: the maximum width of a 1D layered texture.
- maxTexture1DLayers: the maximum number of layers a 1D layered texture can have.

To obtain these values, we:
1. Get the number of CUDA devices with cudaGetDeviceCount.
2. For each device (here we only query device 0 for simplicity), call cudaGetDeviceProperties.
3. Extract the two fields and print them to the console.
4. Handle any errors returned by CUDA API calls.

This program uses only the CUDA runtime API and standard C++ I/O, making it straightforward to compile
with nvcc (e.g., `nvcc -o query_texture_limits query_texture_limits.cu`).
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount returned %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    // For demonstration, query the first device (device 0)
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceProperties returned %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: %s\n", device, prop.name);
    printf("  Maximum 1D layered texture size (width): %d\n", prop.maxTexture1DLayered);
    printf("  Maximum number of layers for 1D layered texture: %d\n", prop.maxTexture1DLayers);

    return 0;
}
```