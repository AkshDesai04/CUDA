```cpp
/*
Write a program to get the ID of the currently active CUDA device using `cudaGetDevice` and print it.

Thinking:
The goal is to query the current CUDA context for the device ID that is set as active. 
CUDA provides the runtime API function `cudaGetDevice(int *device)` which writes the
index of the device that is currently bound to the calling thread. 
We will call this function, check for errors, and print the device ID to stdout.
If an error occurs (e.g., no CUDA-capable device available or the runtime not initialized),
we will print an error message using `cudaGetErrorString` and exit with a non‑zero status.

The program requires the CUDA runtime header `<cuda_runtime.h>` and the standard C
header `<stdio.h>` for printing. 
The `main` function will:
1. Declare an integer to hold the device ID.
2. Call `cudaGetDevice`.
3. Check the returned `cudaError_t`. If it is not `cudaSuccess`, print the error string and return 1.
4. If successful, print the device ID.
5. Return 0.

This simple approach demonstrates how to use `cudaGetDevice` in a standalone
CUDA C program. Compile with `nvcc`:
    nvcc -o get_device_id get_device_id.cu
and run:
    ./get_device_id
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get current CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Current active CUDA device ID: %d\n", device_id);
    return 0;
}
```