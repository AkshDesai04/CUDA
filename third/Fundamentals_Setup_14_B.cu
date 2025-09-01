```cpp
/*
Aim of the program:
Based on the asyncEngineCount, print a message like 'Device has X copy engines.'
If the count is greater than 0, also print 'Data copy can be overlapped with kernel execution.'

Thinking process:
1. Use the CUDA Runtime API to query device properties via cudaGetDeviceProperties.
2. The asyncEngineCount field in cudaDeviceProp tells us how many asynchronous copy engines a device has.
3. The program will:
   - Determine the number of CUDA devices available.
   - For each device, retrieve its properties.
   - Print "Device has X copy engines." where X is asyncEngineCount.
   - If asyncEngineCount > 0, also print "Data copy can be overlapped with kernel execution."
4. Error handling will be minimal for brevity: if cudaGetDeviceProperties fails, print an error and exit.
5. The code is a self‑contained CUDA C++ source file that can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: cudaGetDeviceProperties for device %d failed: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d (%s) has %d copy engines.\n",
               dev, prop.name, prop.asyncEngineCount);

        if (prop.asyncEngineCount > 0) {
            printf("Data copy can be overlapped with kernel execution.\n");
        }
    }

    return 0;
}
```