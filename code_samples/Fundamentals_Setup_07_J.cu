```cpp
/*
Aim of the program:
Check if the device is part of a TCC (Tesla Compute Cluster) (`isTCCDriver`).

Thinking:
The CUDA Runtime API exposes a device attribute called CUDA_DEVICE_ATTRIBUTE_TCC_DRIVER which
returns a non-zero integer if the GPU device is operating in Tesla Compute Cluster (TCC) mode
on Windows. The attribute is queried via cudaDeviceGetAttribute(). The program should
enumerate all CUDA-capable devices present on the system, query this attribute for each
device, and print out whether the device is in TCC mode or not. 
We use the standard cudaGetDeviceCount() to find the number of devices, then loop over
each device index, query the attribute, and print a user-friendly message. Error handling
is performed after each CUDA call to ensure graceful exit in case of failure.

The output will look like:
  Device 0: TCC driver is active.
  Device 1: TCC driver is not active.
If there are no devices, the program will simply state that.

The code is a complete .cu file that can be compiled with nvcc.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        int isTCC = 0;
        CHECK_CUDA(cudaDeviceGetAttribute(&isTCC,
                                          cudaDevAttrTCCDriver,
                                          dev));

        printf("Device %d: ", dev);
        if (isTCC)
            printf("TCC driver is active.\n");
        else
            printf("TCC driver is not active.\n");
    }

    return EXIT_SUCCESS;
}
```