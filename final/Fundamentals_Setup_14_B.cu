```cpp
/*
Based on the `asyncEngineCount`, print a message like 'Device has X copy engines.' 
If the count is greater than 0, also print 'Data copy can be overlapped with kernel execution.'

Thinking:
The goal is to query each CUDA-capable device for its asynchronous copy engine count,
which is stored in the `asyncEngineCount` field of the `cudaDeviceProp` structure.
We will:
1. Determine the number of CUDA devices via cudaGetDeviceCount.
2. Iterate over each device index.
3. Retrieve the device properties with cudaGetDeviceProperties.
4. Print "Device has X copy engines." where X is the asyncEngineCount.
5. If asyncEngineCount > 0, also print the message about overlapping copy with kernel execution.
We will use the CUDA Runtime API and C-style printf for output.
Error checking will be performed for each CUDA call, printing an error message and exiting on failure.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d (%s):\n", dev, prop.name);
        printf("  Device has %d copy engines.\n", prop.asyncEngineCount);

        if (prop.asyncEngineCount > 0) {
            printf("  Data copy can be overlapped with kernel execution.\n");
        }
        printf("\n");
    }

    return 0;
}
```