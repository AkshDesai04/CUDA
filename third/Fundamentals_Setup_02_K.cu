/*
Write a program that creates a summary table, printing the ID, Name, and Global Memory for every CUDA device found.

My approach:
- Include cuda_runtime.h and cstdio.
- In main, call cudaGetDeviceCount to get number of devices.
- If error, print and exit.
- Print table header: "ID\tName\t\tGlobal Memory (GB)".
- For each device ID, use cudaGetDeviceProperties to retrieve struct cudaDeviceProp.
- Convert totalGlobalMem from bytes to gigabytes (divide by (1<<30)) with double precision.
- Print ID, prop.name, and formatted memory. Use printf formatting with width specifiers for alignment.
- Handle case of no devices: print message.
- Return 0.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount returned error code %d: %s\n", err, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Summary of CUDA devices:\n");
    printf("%-5s %-30s %10s\n", "ID", "Name", "Global Mem (GB)");
    printf("------------------------------------------------------------\n");

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties for device %d returned error code %d: %s\n",
                    dev, err, cudaGetErrorString(err));
            continue;
        }
        double memGB = static_cast<double>(prop.totalGlobalMem) / (1ULL << 30);
        printf("%-5d %-30s %10.2f\n", dev, prop.name, memGB);
    }

    return EXIT_SUCCESS;
}
