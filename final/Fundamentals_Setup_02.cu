```c
/* 
Aim of the program: Extend the previous program to iterate through all available CUDA devices. For each device, use `cudaGetDeviceProperties` to retrieve its properties and print its name (e.g., 'NVIDIA GeForce RTX 3080') and its compute capability (e.g., 'Compute Capability: 8.6').

Thinking: 
- We need to include the CUDA runtime header `cuda_runtime.h` and standard I/O.
- First, query the number of devices with `cudaGetDeviceCount`. Handle any error by printing to stderr and exiting.
- If no devices are found, inform the user.
- Iterate over each device index from 0 to deviceCount-1.
- For each, call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` structure.
- On success, print the device index, the device name (`prop.name`), and the compute capability using `prop.major` and `prop.minor`.
- If retrieving properties fails for a device, print an error but continue to the next device.
- Return 0 on success.
- Ensure the program is self‑contained and can compile with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    }

    return 0;
}
```