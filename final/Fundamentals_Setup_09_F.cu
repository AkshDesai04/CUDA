```c
/* 
Aim of the program: Check if the device supports stream priorities (`streamPrioritiesSupported`). 

Thinking:
To determine whether a CUDA-capable device supports stream priorities, we need to query the device attribute
`cudaDevAttrStreamPrioritiesSupported`. This attribute is obtained using `cudaDeviceGetAttribute`. The program
will:
1. Retrieve the number of CUDA devices with `cudaGetDeviceCount`.
2. For each device, query `cudaDevAttrStreamPrioritiesSupported` using `cudaDeviceGetAttribute`.
3. Print whether stream priorities are supported for each device.

We include error checking to ensure CUDA API calls succeed. The output will be printed to standard output.
The program is written in CUDA C (.cu file) and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            return EXIT_FAILURE;                                                \
        }                                                                        \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        int supportsPriorities = 0;
        /* cudaDevAttrStreamPrioritiesSupported is a boolean attribute:
           1 if supported, 0 otherwise */
        CHECK_CUDA(cudaDeviceGetAttribute(&supportsPriorities,
                                          cudaDevAttrStreamPrioritiesSupported,
                                          dev));

        printf("Device %d: Stream priorities supported: %s\n",
               dev, supportsPriorities ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
```