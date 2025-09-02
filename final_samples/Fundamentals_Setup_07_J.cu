```cuda
/*
Aim of the program: Check if the device is part of a TCC (Tesla Compute Cluster) (`isTCCDriver`).

Thinking:
The task is to query each CUDA device to determine whether it is operating in Tesla Compute Cluster (TCC) mode. The CUDA Driver API provides the `cuDeviceGetAttribute` function, which can be used with the attribute `CU_DEVICE_ATTRIBUTE_TCC_DRIVER` to check this status. We need to:
1. Initialize the CUDA driver with `cuInit(0)`.
2. Retrieve the number of devices with `cuDeviceGetCount`.
3. Iterate over each device:
   a. Get the device handle with `cuDeviceGet`.
   b. Query the TCC attribute via `cuDeviceGetAttribute`.
   c. Print whether the device is in TCC mode.
4. Handle any CUDA errors gracefully by printing messages and exiting.
The code will be written in C with CUDA and compiled as a `.cu` file. No external files or libraries beyond the CUDA Driver API are required.*/

#include <stdio.h>
#include <cuda.h>

#define CHECK_CUDA_ERR(err, msg)                                            \
    do {                                                                     \
        if (err != CUDA_SUCCESS) {                                          \
            const char *errStr = nullptr;                                   \
            cuGetErrorString(err, &errStr);                                 \
            fprintf(stderr, "CUDA Error: %s: %s\n", msg,                    \
                    errStr ? errStr : "Unknown error");                     \
            return -1;                                                      \
        }                                                                    \
    } while (0)

int main(void) {
    CUresult res;

    /* Initialize the CUDA driver API */
    res = cuInit(0);
    CHECK_CUDA_ERR(res, "cuInit failed");

    /* Get the number of CUDA devices */
    int deviceCount = 0;
    res = cuDeviceGetCount(&deviceCount);
    CHECK_CUDA_ERR(res, "cuDeviceGetCount failed");

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Detected %d CUDA device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        CUdevice device;
        res = cuDeviceGet(&device, dev);
        CHECK_CUDA_ERR(res, "cuDeviceGet failed");

        /* Retrieve device name */
        char name[256] = {0};
        res = cuDeviceGetName(name, sizeof(name), device);
        CHECK_CUDA_ERR(res, "cuDeviceGetName failed");

        /* Query TCC driver attribute */
        int isTCC = 0;
        res = cuDeviceGetAttribute(&isTCC,
                                   CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
                                   device);
        /* The attribute may not be supported on all GPUs; handle that case */
        if (res == CUDA_ERROR_INVALID_VALUE) {
            printf("Device %d (%s): TCC attribute not supported.\n", dev, name);
            continue;
        }
        CHECK_CUDA_ERR(res, "cuDeviceGetAttribute failed");

        printf("Device %d (%s): TCC mode is %s.\n",
               dev, name, isTCC ? "ENABLED" : "DISABLED");
    }

    return 0;
}
```