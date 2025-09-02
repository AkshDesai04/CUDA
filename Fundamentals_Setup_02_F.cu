```cpp
/*
Write a program that finds the device with the highest compute capability and prints its name and index.

Thinking:
To implement this program we need to query all CUDA devices available on the system. 
We will use `cudaGetDeviceCount` to get the number of devices, then iterate over each device
index, retrieving its properties via `cudaGetDeviceProperties`. The compute capability is
expressed as a pair of integers (major, minor). The highest compute capability is determined
by comparing the major version first, and if those are equal, the minor version.
We will keep track of the device index and name that has the highest compute capability.
After the loop, we print out the index and name of that device. 
Error handling is added through a simple macro that checks CUDA return codes and exits on failure.
The program is self-contained and can be compiled with `nvcc` into a CUDA executable.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",      \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    } while (0)

int main() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    int bestDevice = -1;
    int bestMajor = -1;
    int bestMinor = -1;
    char bestName[256] = {0};

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        if (prop.major > bestMajor ||
            (prop.major == bestMajor && prop.minor > bestMinor)) {
            bestMajor = prop.major;
            bestMinor = prop.minor;
            bestDevice = dev;
            strncpy(bestName, prop.name, sizeof(bestName)-1);
            bestName[sizeof(bestName)-1] = '\0';
        }
    }

    if (bestDevice >= 0) {
        printf("Device with the highest compute capability:\n");
        printf("Index: %d\n", bestDevice);
        printf("Name: %s\n", bestName);
        printf("Compute Capability: %d.%d\n", bestMajor, bestMinor);
    } else {
        printf("Failed to determine the best device.\n");
    }

    return EXIT_SUCCESS;
}
```