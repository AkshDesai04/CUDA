```cpp
/*
Aim: Query and print whether the device can map host memory (`canMapHostMemory` property).

Thinking:
- The goal is to inspect CUDA device capabilities, specifically the flag `canMapHostMemory`
  which indicates whether the device can map host memory (e.g., for pinned memory access).
- To achieve this, the program will:
  1. Include the CUDA runtime header `<cuda_runtime.h>`.
  2. Query the number of CUDA-capable devices using `cudaGetDeviceCount`.
  3. Iterate over each device (or just the first one) to retrieve its properties via
     `cudaGetDeviceProperties`.
  4. Print the value of `props.canMapHostMemory` for each device.
- The program will include basic error checking for CUDA API calls.
- It will be a simple console application that outputs the information to standard output.
- The output will be a single line indicating whether each device can map host memory,
  e.g., "Device 0 can map host memory: 1" (where 1 means true).
- The code will be self‑contained and compile with `nvcc` producing a `.cu` file.

*/

#include <cstdio>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error: cudaGetDeviceProperties for device %d failed: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("  canMapHostMemory: %s\n",
               prop.canMapHostMemory ? "Yes (1)" : "No (0)");
        printf("\n");
    }

    return 0;
}
```