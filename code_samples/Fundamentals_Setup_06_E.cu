```c
/*
Print the maximum y-dimension of a grid of blocks.

The aim of the program is to query the CUDA runtime for the maximum
grid dimensions supported by the current device and then print the
value for the Y‑dimension.  The information is obtained via
cudaGetDeviceProperties which fills a cudaDeviceProp structure.  The
member maxGridSize[1] holds the maximum number of blocks that can be
specified along the Y axis of a grid.  The program handles errors
gracefully by checking the return codes from CUDA API calls and
reporting any failures.  This is a minimal, self‑contained example
suitable for compiling with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaDeviceProp prop;

    // Get device count
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    // Get properties for device 0 (or chosen device)
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum Y dimension of a grid
    printf("Maximum Y-dimension of a grid of blocks: %d\n", prop.maxGridSize[1]);

    return 0;
}
```