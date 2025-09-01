```cuda
/* 
Aim of the program:
Write a program to determine if the primary GPU is an integrated (sharing memory with the CPU) or a discrete device. Print the result.

Thinking:
The CUDA Runtime API provides the function cudaGetDeviceProperties() which fills a cudaDeviceProp structure with information about a given GPU. This structure contains an `int integrated` member: it is set to 1 if the GPU is integrated (shares memory with the CPU), and 0 for a discrete device. 

The program steps are:
1. Check that at least one CUDA-capable device exists using cudaGetDeviceCount().
2. Retrieve the properties of device 0 (the primary GPU) via cudaGetDeviceProperties().
3. Inspect the `integrated` field of the retrieved properties.
4. Print "Integrated GPU" if `integrated` is 1, otherwise print "Discrete GPU".
5. Handle any CUDA runtime errors gracefully, printing an error message and exiting with a non‑zero status.

This implementation uses only the CUDA Runtime API and is portable across CUDA‑enabled systems.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0); // primary GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (prop.integrated) {
        printf("Integrated GPU\n");
    } else {
        printf("Discrete GPU\n");
    }

    return 0;
}
```