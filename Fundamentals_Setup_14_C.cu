```cuda
/*
Aim of the program: 
Iterate through all devices on the system and print the `asyncEngineCount` for each.

Thought process:
1. The CUDA Runtime API provides functions to query the number of devices (`cudaGetDeviceCount`) and to retrieve properties for each device (`cudaGetDeviceProperties`).
2. The `cudaDeviceProp` structure contains a member `asyncEngineCount`, which indicates the number of asynchronous engine units on a GPU.
3. The program should:
   a. Query the total device count.
   b. Iterate over each device index.
   c. Retrieve the properties for that device.
   d. Print the device index, name, and asyncEngineCount.
4. Error handling is important: after each CUDA call, check the returned `cudaError_t` and print a meaningful message if an error occurs.
5. Use `printf` from the C standard library for output, as the program is in C/CUDA and should compile as a `.cu` file.
6. The program should be minimal, self-contained, and compile with `nvcc`.

Implementation details:
- Include `<cuda_runtime.h>` for CUDA runtime API and `<stdio.h>` for printing.
- In `main()`, first get the device count.
- If no devices are found, print a message and exit gracefully.
- Loop over each device, get its properties, and print the desired information.
- Return 0 on success, -1 on error.

The final code follows this plan and is ready to compile as a CUDA source file (.cu).
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Detected %d CUDA device(s):\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            printf("cudaGetDeviceProperties failed for device %d: %s\n",
                   dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  asyncEngineCount = %d\n", prop.asyncEngineCount);
    }

    return 0;
}
```