```cpp
/* 
Aim of the program: 
Look up the documentation for `cudaErrorInvalidDevice`. What other functions can return this error?

Thinking:
The goal is to demonstrate that the CUDA error code `cudaErrorInvalidDevice` can be produced by several API calls when an invalid device index is supplied. Typical functions that accept a device ID include:
- cudaSetDevice
- cudaGetDeviceProperties
- cudaDeviceReset
- cudaDeviceGetAttribute
- cudaGetDevice (though this retrieves the current device, passing an invalid pointer might indirectly cause issues)
- cudaDeviceGetPCIBusId (when combined with an invalid device)

To illustrate this, the program will:
1. Attempt to set an invalid device index using `cudaSetDevice`.
2. Query properties for an invalid device using `cudaGetDeviceProperties`.
3. Reset an invalid device using `cudaDeviceReset`.
4. Retrieve an attribute for an invalid device using `cudaDeviceGetAttribute`.
Each call will print the returned error code and its human-readable string. This demonstrates that all these functions can return `cudaErrorInvalidDevice` when given an out‑of‑range device ID. */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int invalidDevice = 999;  // Intentionally out of range

    // 1. cudaSetDevice with invalid device
    cudaError_t err = cudaSetDevice(invalidDevice);
    printf("cudaSetDevice(%d) returned: %s (%d)\n",
           invalidDevice, cudaGetErrorString(err), err);

    // 2. cudaGetDeviceProperties with invalid device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, invalidDevice);
    printf("cudaGetDeviceProperties(&prop, %d) returned: %s (%d)\n",
           invalidDevice, cudaGetErrorString(err), err);

    // 3. cudaDeviceReset with invalid device
    // First set a valid device so that reset is meaningful
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount > 0) {
        cudaSetDevice(0);
    }
    err = cudaDeviceReset();
    // Note: cudaDeviceReset operates on the current device, so if we had an invalid device set
    // earlier, this would return cudaErrorInvalidDevice. Since we reset to a valid device,
    // this should succeed. To force an error, we deliberately set an invalid device again.
    cudaSetDevice(invalidDevice);
    err = cudaDeviceReset();
    printf("cudaDeviceReset() after setting invalid device returned: %s (%d)\n",
           cudaGetErrorString(err), err);

    // 4. cudaDeviceGetAttribute with invalid device
    int attribute = 0;
    err = cudaDeviceGetAttribute(&attribute,
                                 cudaDevAttrMaxThreadsPerBlock,
                                 invalidDevice);
    printf("cudaDeviceGetAttribute(..., %d) returned: %s (%d)\n",
           invalidDevice, cudaGetErrorString(err), err);

    return 0;
}
```