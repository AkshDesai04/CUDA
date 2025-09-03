```cpp
/*
Aim: Query and print whether the device supports stream priorities.

Thinking:
To determine whether the CUDA-capable device supports stream priorities, we need to query a device attribute
provided by the CUDA Runtime API. The attribute of interest is cudaDevAttrStreamPrioritiesSupported.
The runtime API function cudaDeviceGetAttribute() retrieves the value of a device attribute.
The value will be 1 if stream priorities are supported, 0 otherwise.

Steps:
1. Include necessary headers: <cuda_runtime.h> for CUDA API and <cstdio> for printing.
2. In main, we first query the number of CUDA devices using cudaGetDeviceCount().
   If there are no devices, we print an error and exit.
3. We select device 0 (or the current device) for simplicity.
4. Call cudaDeviceGetAttribute() with attribute cudaDevAttrStreamPrioritiesSupported.
   Check the returned status; if it fails, report the error.
5. Based on the attribute value, print "Stream priorities are supported." or "Stream priorities are NOT supported."
6. Return 0 on success.

This program does not launch any kernels; it only queries device properties and prints the result.
It can be compiled with nvcc: nvcc -o stream_priority_check stream_priority_check.cu
*/

#include <cuda_runtime.h>
#include <cstdio>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return 1;
    }

    // Choose device 0 for this query
    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int supportsStreamPriorities = 0;
    err = cudaDeviceGetAttribute(&supportsStreamPriorities,
                                 cudaDevAttrStreamPrioritiesSupported,
                                 device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (supportsStreamPriorities) {
        printf("Stream priorities are supported on device %d.\n", device);
    } else {
        printf("Stream priorities are NOT supported on device %d.\n", device);
    }

    return 0;
}
```