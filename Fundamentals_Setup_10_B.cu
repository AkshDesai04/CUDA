/*
Aim: Iterate through all available GPUs and print the L2 cache size for each.

Thinking:
The task requires a CUDA program written in C that enumerates all GPUs present on the system and reports the size of their L2 cache. 
In CUDA, the `cudaGetDeviceCount` function returns the number of GPUs. For each device index, `cudaGetDeviceProperties` fills a `cudaDeviceProp` structure that contains various hardware attributes, including `l2CacheSize` which is an integer representing the size in bytes. 

The program will:
1. Include the necessary headers (`stdio.h` and `cuda_runtime.h`).
2. Call `cudaGetDeviceCount` to find out how many GPUs exist.
3. Loop over each device index, retrieve its properties, and print the `l2CacheSize`.
4. Handle the case where no GPUs are found by printing an appropriate message.
5. Perform minimal error checking by verifying the return codes of CUDA API calls and reporting any failures.

The output will be a simple console message per GPU:
```
GPU <index>: L2 Cache Size = <size> bytes
```
If no GPUs are present, the program will output:
```
No CUDA-compatible GPUs found.
```
The code is written to be compiled with `nvcc` and is self-contained in a single `.cu` file. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count (error %d: %s)\n", err, cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-compatible GPUs found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get properties for device %d (error %d: %s)\n",
                    dev, err, cudaGetErrorString(err));
            continue;
        }

        printf("GPU %d: L2 Cache Size = %d bytes\n", dev, prop.l2CacheSize);
    }

    return 0;
}
