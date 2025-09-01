/*
Query and print the maximum dimension of a 1D texture for device 0.

Thinking:
- The goal is to retrieve the maximum size supported for 1D textures on CUDA device 0.
- CUDA runtime provides a struct cudaDeviceProp via cudaGetDeviceProperties().
- The struct contains a field maxTexture1D that holds this value.
- We'll set device 0, call cudaGetDeviceProperties, then print the maxTexture1D value.
- Include necessary headers: <cuda_runtime.h> and <stdio.h>.
- Handle errors by checking return values of CUDA API calls.
- The program will simply output the value to stdout.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaDeviceProp prop;

    // Query device properties for device 0
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                device, cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum dimension of a 1D texture
    printf("Maximum 1D texture dimension on device %d: %u\n", device, prop.maxTexture1D);

    return 0;
}
