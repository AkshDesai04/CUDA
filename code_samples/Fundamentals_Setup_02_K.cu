/*
Aim: Write a program that creates a summary table, printing the ID, Name, and Global Memory for every CUDA device found.

Thinking:
1. We need to query the CUDA runtime for the number of devices using cudaGetDeviceCount.
2. Iterate over each device index from 0 to count-1.
3. For each device, retrieve its properties via cudaGetDeviceProperties.
4. From the properties structure we can access:
   - name (char array) for the device name.
   - totalGlobalMem (size_t) for the total global memory in bytes.
5. Print a header row and then a formatted line for each device:
   - ID as an integer.
   - Name left-aligned with a reasonable width.
   - Global memory displayed in bytes, or converted to GB for readability.
6. Include basic error handling: if no devices are found, print a message.
7. Ensure the program is a valid CUDA C source file (.cu) that compiles with nvcc.
8. No external dependencies beyond <cuda_runtime.h> and standard headers.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("%-3s %-30s %15s\n", "ID", "Name", "Global Memory (GB)");
    printf("---------------------------------------------------------------\n");

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        double memGB = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        printf("%-3d %-30s %14.2f\n", dev, prop.name, memGB);
    }

    return 0;
}
