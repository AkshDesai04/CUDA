/*
Write a program that creates a summary table, printing the ID, Name, and Global Memory for every CUDA device found.

My thinking:
- The goal is to enumerate all CUDA-capable devices on the system.
- Use the CUDA Runtime API: cudaGetDeviceCount to get the number of devices.
- For each device ID, call cudaGetDeviceProperties to retrieve the struct cudaDeviceProp.
- The struct contains fields such as name (char[256]) and totalGlobalMem (size_t).
- We'll format the output as a simple table with columns: ID, Name, Global Memory (in bytes, maybe convert to GB for readability).
- Use printf for formatting.
- Add basic error handling: if cudaGetDeviceCount fails, print error and exit.
- Use standard C includes: <cstdio> for printf, <cuda_runtime.h> for CUDA API.
- The program is a single .cu file; it can be compiled with nvcc.
- The main function will return 0 on success.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("%-4s %-30s %-15s\n", "ID", "Name", "Global Memory (GB)");
    printf("------------------------------------------------------------\n");

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        // Convert bytes to gigabytes for readability
        double memGB = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        printf("%-4d %-30s %-15.2f\n", dev, prop.name, memGB);
    }

    return 0;
}
