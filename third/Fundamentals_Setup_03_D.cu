```cuda
/*
Aim of the program: Query and print the memory bus width of the primary GPU in bits.

Thought Process:
1. The CUDA runtime API provides a way to query device properties via cudaGetDeviceProperties.
2. The cudaDeviceProp structure contains a field named memoryBusWidth which holds the width of the GPU's memory bus in bits.
3. We need to:
   - Select the primary GPU (device 0).
   - Retrieve its properties.
   - Print the memoryBusWidth value.
4. Error handling: After each CUDA API call, check the returned cudaError_t and report any failures.
5. The program will be a simple console application that compiles to a .cu file, uses the standard CUDA headers, and prints the bus width to stdout.
6. The output should be purely the integer value (bits) followed by a newline.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int device = 0; // Primary GPU
    cudaDeviceProp prop;
    cudaError_t err;

    // Get properties for device 0
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the memory bus width in bits
    printf("%d\n", prop.memoryBusWidth);

    return 0;
}
```