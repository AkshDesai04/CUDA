/*
Aim: Query and print the texture alignment requirement for device 0.

Thinking:
- To obtain device-specific information in CUDA, we use cudaGetDeviceProperties.
- The cudaDeviceProp structure contains a member textureAlignment which
  specifies the required alignment for texture memory accesses in bytes.
- We'll query properties for device 0, check for errors, and then print
  the textureAlignment value.
- A simple main function suffices; no kernel launches are needed.
- Error handling is minimal: if cudaGetDeviceProperties fails, print
  an error message and exit with a non-zero status.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaDeviceProp prop;

    /* Query device properties */
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                device, cudaGetErrorString(err));
        return -1;
    }

    /* Print the texture alignment requirement */
    printf("Texture alignment requirement for device %d: %zu bytes\n",
           device, prop.textureAlignment);

    return 0;
}
