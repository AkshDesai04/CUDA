```c
/*
Aim of the program: Query and print the maximum y-dimension of a thread block.

Thinking:
The task requires a simple CUDA host program that queries device properties
and prints the maximum number of threads that can be launched in the Y
dimension of a thread block. The CUDA runtime API provides a structure
`cudaDeviceProp` that contains this information in the `maxThreadsDim[3]`
array, where index 1 corresponds to the Y dimension.

The program will:
1. Use `cudaGetDeviceCount` to ensure at least one CUDA device is present.
2. Retrieve the properties of device 0 using `cudaGetDeviceProperties`.
3. Extract `prop.maxThreadsDim[1]` and print it.
4. Handle any errors by printing descriptive messages and returning a
   non‑zero exit code.

The code is minimal and self‑contained, written in C and suitable for
compilation with `nvcc`.
*/
#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n");
        return 1;
    }

    int device = 0; // Use the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum y-dimension of a thread block: %d\n", prop.maxThreadsDim[1]);

    return 0;
}
```