```cuda
/*
Aim: Query and print the maximum y-dimension of a thread block.

Thinking:
To obtain the maximum y-dimension of a thread block supported by the CUDA device, we use the CUDA Runtime API function `cudaGetDeviceProperties`. This function fills a `cudaDeviceProp` structure with various properties of the device, including `maxThreadsDim`, which holds the maximum number of threads per block in each dimension (x, y, z). 
We retrieve the properties for the current device (obtained via `cudaGetDevice`), then print the `maxThreadsDim.y` value. Basic error checking is performed: if retrieving the device properties fails, an error message is printed and the program exits with a non-zero status. 
The program is simple, self-contained, and can be compiled with `nvcc` and run on any CUDA-capable GPU.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum thread block y-dimension for device %d (%s): %d\n",
           device, prop.name, prop.maxThreadsDim.y);

    return 0;
}
```