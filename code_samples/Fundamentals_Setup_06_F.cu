```cuda
/*
Aim: Print the maximum z-dimension of a grid of blocks.

Thinking:
- CUDA devices expose their capabilities via the struct cudaDeviceProp.
- The field `maxGridSize` is an array of three integers that specify the
  maximum number of blocks in the x, y, and z dimensions of a grid
  that can be launched on that device.
- We can query the properties of the current device using
  `cudaGetDeviceProperties`.
- After retrieving the struct, the maximum z-dimension is simply
  `maxGridSize[2]`.
- The program will print this value to stdout.
- Basic error handling is performed to ensure `cudaGetDeviceProperties`
  succeeds.
- The code is self‑contained and can be compiled with `nvcc` and run
  on a CUDA capable machine.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int dev = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device (error code %d: %s)!\n",
                err, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties (error code %d: %s)!\n",
                err, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Maximum z-dimension of a grid of blocks: %d\n", prop.maxGridSize[2]);

    return EXIT_SUCCESS;
}
```