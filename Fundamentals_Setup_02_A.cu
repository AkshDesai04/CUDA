```cuda
/*
Aim of the program:
Iterate through all available CUDA devices and print their total global memory in gigabytes (GiB) for each.

Thinking:
1. The goal is to discover all CUDA-capable devices present in the system and retrieve the amount of global memory each possesses.
2. We will use the CUDA Runtime API:
   - cudaGetDeviceCount() to find out how many devices exist.
   - For each device index, cudaGetDeviceProperties() to obtain a cudaDeviceProp structure that contains totalGlobalMem (bytes).
3. Convert totalGlobalMem from bytes to GiB. GiB is 2^30 bytes (i.e., 1024^3).
4. Print the device number, name, and memory in a human-readable format.
5. Handle errors: if cudaGetDeviceCount or cudaGetDeviceProperties fails, we should report the error and exit gracefully.
6. The program should compile as a standard CUDA C++ source file (.cu) and can be built with `nvcc` or any CUDA-aware compiler.
7. We’ll include <cuda_runtime.h> and <cstdio> for the CUDA runtime API and C-style I/O.
8. Since the program is simple and purely CPU-side, we don't need any GPU kernels.
9. We'll use `printf` for output, which works in host code. We also format the GiB value to, say, two decimal places for readability.
10. Return 0 on success.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        // totalGlobalMem is in bytes; convert to GiB
        double memGiB = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Total Global Memory: %.2f GiB\n", memGiB);
    }

    return 0;
}
```