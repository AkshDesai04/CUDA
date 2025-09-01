/*
Query and print the maximum dimensions (width and height) of a 2D texture for device 0.

The goal of this program is to retrieve and display the maximum 2D texture
size supported by the first CUDA-capable GPU (device 0).  
To achieve this we use the CUDA runtime API:  
1. Declare a cudaDeviceProp structure.  
2. Call cudaGetDeviceProperties() with device ID 0 to fill the structure.  
3. Access the maxTexture2D array which contains two integers: the maximum
   width (index 0) and height (index 1).  
4. Print these values to standard output.  

No kernels or GPU memory allocation is needed – all work is performed on the host side.
Error checking is included to handle cases where cudaGetDeviceProperties fails.  
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    cudaDeviceProp prop;
    int device = 0;

    // Retrieve device properties for device 0
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get properties for device %d: %s\n",
                device, cudaGetErrorString(err));
        return 1;
    }

    // Print maximum 2D texture dimensions
    printf("Maximum 2D texture size for device %d:\n", device);
    printf("  Width  : %d\n", prop.maxTexture2D[0]);
    printf("  Height : %d\n", prop.maxTexture2D[1]);

    return 0;
}
