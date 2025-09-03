```cpp
/* 
Query and print the maximum dimensions (width and height) of a 2D texture for device 0.
Thinking:
- Use the CUDA Runtime API to set the device to 0.
- Retrieve device properties via cudaGetDeviceProperties.
- The struct cudaDeviceProp contains a member maxTexture2D of type cudaTextureAlignment, which has width and height fields.
- After fetching properties, print the width and height using printf.
- Add basic error checking for CUDA API calls.
- Provide a minimal main() that performs these steps and exits.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum 2D texture dimensions for device %d:\n", device);
    printf("  Width  = %u\n", prop.maxTexture2D.width);
    printf("  Height = %u\n", prop.maxTexture2D.height);

    return 0;
}
```