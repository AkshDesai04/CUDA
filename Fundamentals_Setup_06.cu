```c
/*
Aim of the program: Query and print the maximum dimensions (x, y, z) of a thread block for the primary CUDA device.
The result should be displayed clearly, for example: 'Max Block Dimensions: [X, Y, Z]'.

Thinking:
1. Include necessary headers: <stdio.h> and <cuda_runtime.h>.
2. Obtain the current CUDA device using cudaGetDevice().
3. Allocate a cudaDeviceProp structure and fill it with cudaGetDeviceProperties().
4. Extract the maxThreadsDim.x, maxThreadsDim.y, and maxThreadsDim.z fields from the structure.
5. Print them in the required format.
6. Perform basic error checking after CUDA calls and exit with non-zero status if any error occurs.
7. Ensure the program compiles as a .cu file and uses the CUDA runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;  // Primary device
    cudaDeviceProp prop;

    // Get current device (should be device 0 by default)
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Get device properties
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print maximum block dimensions
    printf("Max Block Dimensions: [%d, %d, %d]\n",
           prop.maxThreadsDim.x,
           prop.maxThreadsDim.y,
           prop.maxThreadsDim.z);

    return 0;
}
```