```cuda
/*
Aim of the program:
Check if the device is part of a TCC (Tesla Compute Cluster) (isTCCDriver).

Thinking:
To determine whether a CUDA-capable GPU is running under a Tesla Compute Cluster (TCC) driver, we can query the device properties provided by the CUDA Runtime API. Each device property struct (`cudaDeviceProp`) contains a boolean field `tccDriver`. When this flag is true, the device is operating in TCC mode (common on Windows Server installations for Tesla GPUs). The program will:
1. Get the total number of CUDA devices via `cudaGetDeviceCount`.
2. For each device, retrieve its properties using `cudaGetDeviceProperties`.
3. Print the device index and the value of the `tccDriver` flag.
The program uses simple error checking to ensure CUDA calls succeed. It is a standalone `.cu` file that can be compiled with `nvcc` and run on any system with CUDA installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple macro for CUDA error checking */
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1;                                            \
        }                                                          \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Detected %d CUDA device(s).\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("Device %d: \"%s\"\n", i, prop.name);
        printf("  TCC Driver: %s\n", prop.tccDriver ? "Yes" : "No");
        printf("\n");
    }

    return 0;
}
```