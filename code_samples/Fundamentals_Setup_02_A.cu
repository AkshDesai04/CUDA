/*
Iterate through all available CUDA devices and print their total global memory in gigabytes (GiB) for each.

Thinking: The goal is to write a simple CUDA host program that enumerates all CUDA-capable devices present in the system and reports each device's total global memory in GiB. Since no device-side kernels are needed, the program will simply use the CUDA runtime API. The steps are:
1. Include the necessary headers: <cuda_runtime.h> for CUDA API and <cstdio> for printing.
2. Call cudaGetDeviceCount to obtain the number of CUDA devices.
3. If no devices are found, print an informative message and exit.
4. Iterate over each device index.
5. For each index, call cudaGetDeviceProperties to get a cudaDeviceProp structure.
6. Convert the `totalGlobalMem` field from bytes to GiB by dividing by (1024^3). Use a double for precision.
7. Print the device index, name, and the memory value formatted to two decimal places.
8. Handle any CUDA errors returned by API calls gracefully (though for this simple program basic error checking is sufficient).
9. Return 0 from main.

No kernel launches or device code are required, so this file can be compiled with nvcc as a standard host-only program.
*/

#include <cuda_runtime.h>
#include <cstdio>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error retrieving properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        double memGiB = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Total Global Memory: %.2f GiB\n", memGiB);
    }

    return 0;
}
