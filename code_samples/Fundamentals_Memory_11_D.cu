```cpp
/*
What is the size limit for constant memory? (Typically 64KB).

This program queries the CUDA device properties to obtain the actual
size of the constant memory for the first GPU available. The
cudaDeviceProp structure contains a field constMemBytes which
stores the number of bytes of constant memory available on the
device. The program prints this value in bytes and in kilobytes
to show the actual limit. This demonstrates how the limit can be
queried programmatically, and it verifies that most modern GPUs
report a constant memory size of 65536 bytes (64 KB). 
Additionally, the program enumerates all devices and prints the
constant memory size for each one to provide a complete picture
of the available constant memory across the system.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable device found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Constant memory size: %zu bytes (%.2f KB)\n",
               prop.constMemBytes,
               static_cast<double>(prop.constMemBytes) / 1024.0);
    }

    return 0;
}
```