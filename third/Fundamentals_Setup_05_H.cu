/*
Aim: Query and print the maximum y-dimension of a thread block.

Thinking:
The CUDA runtime API provides device properties that include the maximum block dimensions in the x, y, and z directions. These can be accessed via cudaGetDeviceProperties, which fills a cudaDeviceProp structure. The field maxThreadsDim is an array where index 0 corresponds to the maximum number of threads in the X dimension, index 1 to Y, and index 2 to Z. Therefore, to obtain the maximum Y dimension, we retrieve this property and print maxThreadsDim[1].

We will:
1. Include <stdio.h> for printing.
2. Call cudaGetDeviceProperties for the current device (device 0 is sufficient for most systems).
3. Check for errors from CUDA calls.
4. Print the maximum Y dimension.

This is a standalone CUDA C program and can be compiled with `nvcc max_y_dim.cu -o max_y_dim` and run to see the result. No GPU kernel is launched; we only query device information.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // Default device
    cudaDeviceProp prop;
    cudaError_t err;

    // Get device properties
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties (error code %d): %s\n",
                err, cudaGetErrorString(err));
        return 1;
    }

    // Print the maximum y-dimension of a thread block
    printf("Maximum Y dimension of a thread block on device %d: %d\n",
           device, prop.maxThreadsDim[1]);

    return 0;
}
