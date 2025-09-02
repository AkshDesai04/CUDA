```cpp
/*
Aim: Write a program to determine if the primary GPU is an integrated (sharing memory with the CPU) or a discrete device. Print the result.

Thinking:
1. The goal is to inspect the properties of the primary CUDA device and decide whether it is an integrated GPU or a discrete GPU.
2. CUDA exposes device properties via the `cudaDeviceProp` struct, which includes a boolean field `integrated`. This field is true if the GPU shares memory with the host (i.e., an integrated GPU).
3. To get the properties of the primary device, we first query the current device ID using `cudaGetDevice()`. Then we retrieve the properties with `cudaGetDeviceProperties()`.
4. We will check the `integrated` flag:
   - If `integrated` is true, print "Integrated GPU".
   - Otherwise, print "Discrete GPU".
5. The program will handle any CUDA errors gracefully by printing an error message and exiting with a non-zero status.
6. The code will be written in standard C++ with CUDA extensions and compiled with `nvcc`. It uses `printf` for output.

This approach directly uses the CUDA runtime API, so no external libraries or complex logic are required.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceId = 0;
    cudaError_t err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess)
    {
        printf("Error getting current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess)
    {
        printf("Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (prop.integrated)
    {
        printf("Integrated GPU\n");
    }
    else
    {
        printf("Discrete GPU\n");
    }

    return 0;
}
```