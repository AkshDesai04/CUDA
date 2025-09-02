```cuda
/*
Aim of the program:
Declare a `__device__` variable in a header file. Does this cause problems? (Yes, can lead to multiple definitions unless handled with `extern`).

Thinking:
- A __device__ variable lives in GPU memory and can be accessed from any kernel.
- If the header contains a full definition (e.g., `__device__ int devVar = 0;`) and is included in multiple .cu files, each translation unit will have its own copy of the variable. During linking, the linker will see multiple definitions of the same symbol and report an error.
- The standard solution is to put only an extern declaration in the header (`extern __device__ int devVar;`) and provide a single definition in exactly one .cu file (`__device__ int devVar = 0;`).
- The code below demonstrates this pattern: the header contains the extern declaration, this .cu file contains the definition, and a simple kernel reads the variable. If you were to uncomment the definition in the header and compile more than one .cu file, you would see the multiple‑definition error.
*/

#include <stdio.h>
#include "device_var.h"   // This header declares: extern __device__ int deviceVar;

// Definition of the device variable (only in one compilation unit)
__device__ int deviceVar = 42;

// Kernel that prints the value of the device variable
__global__ void printDeviceVar()
{
    // Note: printf from device code requires device printf support
    printf("deviceVar = %d\n", deviceVar);
}

int main()
{
    // Launch kernel
    printDeviceVar<<<1, 1>>>();

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```