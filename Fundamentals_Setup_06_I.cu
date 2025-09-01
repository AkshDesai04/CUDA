```cuda
/* 
Query and print the maximum layered 1D texture size and number of layers.

In this program we aim to retrieve the CUDA device properties that specify
the limits for 1‑dimensional layered textures.  The CUDA driver API exposes
these limits in the cudaDeviceProp structure, specifically:
  - `maxTexture1DLayered`: the maximum width (in texels) of a 1‑D layered
    texture.
  - `maxTexture1DLayeredDepth`: the maximum number of layers that can be
    packed into such a texture.

The steps we take are:
1. Determine how many CUDA devices are present (`cudaGetDeviceCount`).
2. For the first device (index 0) retrieve its properties via
   `cudaGetDeviceProperties`.
3. Extract and print the two relevant fields.
4. Handle any CUDA errors by printing a message and exiting.

This program is minimal and portable across CUDA versions that expose
these properties (CUDA 4.0 and later).  It demonstrates how a host
application can query hardware capabilities without launching any kernels.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum layered 1D texture width (size): %d\n", prop.maxTexture1DLayered);
    printf("Maximum number of layers for 1D layered textures: %d\n", prop.maxTexture1DLayeredDepth);

    return 0;
}
```