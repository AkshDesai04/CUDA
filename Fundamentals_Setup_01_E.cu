```cuda
/*
Aim of the program:
Query and print the memory clock rate of the primary CUDA device in kilohertz (kHz).

Thinking:
- The CUDA Runtime API provides cudaGetDeviceProperties which fills a cudaDeviceProp struct.
- The struct contains a member `memoryClockRate` that is reported in kilohertz.
- To target the primary device, we assume device ID 0 is the primary.
- We will perform basic error checking: ensure a CUDA device is present, and the query succeeds.
- The program will print the memory clock rate directly, as it is already in kHz.
- No kernel launches are required; this is a simple host-side query.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    int deviceId = 0; // primary device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Memory Clock Rate of device %d (%s): %d kHz\n",
           deviceId, prop.name, prop.memoryClockRate);

    return 0;
}
```