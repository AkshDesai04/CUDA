/*
Write a program that gets the driver version and prints its major and minor versions separately. (e.g., Driver version 11.4 -> Major: 11, Minor: 4).

I started by interpreting the requirement: we need a CUDA program that queries the installed CUDA driver version and extracts the major and minor components. The CUDA runtime API provides a function called `cudaDriverGetVersion` which returns the driver version as an integer. The encoding used by CUDA is that the major version is the integer division by 100, and the minor version is the remainder when dividing by 100. For example, a driver version of 1104 corresponds to major 11 and minor 4.

To implement this, I plan to:
1. Include the standard I/O header and the CUDA runtime header.
2. Call `cudaDriverGetVersion`, checking for errors.
3. Compute the major and minor components.
4. Print them out in a clear format.

I also decided to provide a simple error check in case `cudaDriverGetVersion` fails, printing the error message. The code is straightforward and can be compiled with `nvcc` as a standard CUDA C source file.

Below is the full .cu file containing the program.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int driver_version = 0;
    cudaError_t err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA driver version: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int major = driver_version / 100;
    int minor = driver_version % 100;

    printf("Driver version: %d.%d\n", major, minor);
    printf("Major: %d\n", major);
    printf("Minor: %d\n", minor);

    return 0;
}
