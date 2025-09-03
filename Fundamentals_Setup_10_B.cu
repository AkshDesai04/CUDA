/*
Aim of the program: Iterate through all available GPUs and print the L2 cache size for each.

My thinking process:
1. We need to query the number of CUDA-capable devices on the system. The CUDA runtime API provides `cudaGetDeviceCount`.
2. For each device index, we can retrieve detailed properties with `cudaGetDeviceProperties`. The `cudaDeviceProp` struct includes a field `l2CacheSize` that reports the L2 cache size in bytes.
3. We should handle cases where there are no devices or an error occurs. For each error from the CUDA API, we can print an informative message and exit with a non-zero status.
4. The program will print the device index, device name, and the L2 cache size in bytes. Optionally, we could format the size into KB or MB for readability, but the requirement only asks for the size, so bytes are fine.
5. We'll use `printf` from `<cstdio>` for output. The program is straightforward and self-contained in a single .cu file, suitable for compilation with `nvcc`.
6. To ensure portability, we include minimal headers and use `cudaError_t` for error checking. The program will exit with `EXIT_FAILURE` if any CUDA call fails.
7. The final code will have a `main` function, perform the queries, loop over devices, and print results. No GPU kernels or device code are required, just host code.

*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        // The L2 cache size is reported in bytes. If the device does not report it,
        // the field may be zero. We'll still print the value.
        printf("Device %d: %s\n", dev, prop.name);
        printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
    }

    return EXIT_SUCCESS;
}
