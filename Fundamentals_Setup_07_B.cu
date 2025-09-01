/*
Aim of the program: 
Given the maximum grid dimensions, calculate the theoretical maximum number of blocks you could launch in total (though this is often limited by other factors).

Thinking:
- In CUDA, each device reports a maximum grid size in each dimension (x, y, z) through the cudaDeviceProp structure's maxGridSize array.
- A kernel launch can specify a 3‑D grid where each dimension can be up to the corresponding maxGridSize value.
- The theoretical maximum number of blocks for a single launch is the product of these three limits: 
      maxBlocks = maxGridSize[0] * maxGridSize[1] * maxGridSize[2]
- Actual limits may be lower due to resource constraints, but this product gives the absolute upper bound according to the device specs.
- The program will:
  1. Query the current device (default 0).
  2. Retrieve the maxGridSize values.
  3. Compute the product.
  4. Print each dimension and the computed maximum.
- We'll include basic error checking for CUDA API calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Device %d: %s\n", device, prop.name);
    printf("Maximum grid dimensions (x, y, z): %d, %d, %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    long long maxBlocks = (long long)prop.maxGridSize[0] *
                          (long long)prop.maxGridSize[1] *
                          (long long)prop.maxGridSize[2];

    printf("Theoretical maximum number of blocks that can be launched in a single kernel launch: %lld\n", maxBlocks);

    return 0;
}
