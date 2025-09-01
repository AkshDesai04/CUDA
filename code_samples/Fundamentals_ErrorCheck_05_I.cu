/*
Aim: Trigger `cudaErrorNoDevice` by trying to run a CUDA program on a system with no NVIDIA driver or no CUDA-capable GPU.

Thinking:
The goal is to intentionally provoke the CUDA runtime error `cudaErrorNoDevice`, which is returned when the runtime detects that no CUDA-capable device is present on the system. A simple and reliable way to generate this error is to query the number of CUDA devices using `cudaGetDeviceCount`. On a machine without an NVIDIA driver or GPU, this function returns `cudaErrorNoDevice`. Once the error is obtained, we print the error string so that the user can see the message.

This program:
1. Includes the necessary header `<stdio.h>` for printing.
2. Calls `cudaGetDeviceCount` to retrieve the number of devices.
3. Checks the returned `cudaError_t`. If the error is not `cudaSuccess`, it prints the error message and exits.
4. If, unexpectedly, a device is found, it prints the count and exits successfully.

By compiling and running this code on a system lacking CUDA support, the program will output an error message indicating `cudaErrorNoDevice`, satisfying the requirement.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);
    return 0;
}
