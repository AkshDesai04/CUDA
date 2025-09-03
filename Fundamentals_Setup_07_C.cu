```cpp
/*
Write a program that prints a summary of the execution hierarchy limits: Max Grid [X, Y, Z] and Max Block [X, Y, Z].

I want to provide a concise CUDA host program that queries and displays the maximum grid and block dimensions supported by the GPU(s). 
The plan is to:
1. Include the necessary headers: <stdio.h> for I/O and <cuda_runtime.h> for CUDA runtime API.
2. Use cudaGetDeviceCount to determine how many GPUs are present.  
3. For each device, call cudaGetDeviceProperties to obtain the cudaDeviceProp structure, which contains:
   - maxGridSize[3]      → maximum grid dimensions in X, Y, Z.
   - maxThreadsDim[3]   → maximum block dimensions in X, Y, Z.
4. Print these values in a readable format.  
5. Perform basic error checking after each CUDA API call and exit gracefully if something fails.

The program is written to be compiled with `nvcc` and run on a machine that has CUDA installed. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Max Grid Dimensions (X, Y, Z): [%d, %d, %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max Block Dimensions (X, Y, Z): [%d, %d, %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("\n");
    }

    return EXIT_SUCCESS;
}
```