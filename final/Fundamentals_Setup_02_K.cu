```c
/*
Write a program that creates a summary table, printing the ID, Name, and Global Memory for every CUDA device found.

Thinking:
- The program needs to enumerate all CUDA-capable devices present on the system.
- We will use the CUDA Runtime API: cudaGetDeviceCount() to find how many devices exist.
- For each device, cudaGetDeviceProperties() gives a struct cudaDeviceProp containing
  the device name (char name[256]) and the amount of global memory (size_t totalGlobalMem).
- We'll output a simple table with headers: ID, Name, Global Memory (bytes or GB).
- Error handling: if cudaGetDeviceCount returns 0 we print a message; if any CUDA call fails,
  we print an error message and exit.
- Formatting: use printf with a width for columns to keep the table readable.
- The program will compile with nvcc and produce an executable that, when run, prints the table.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (device_count == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("%-5s %-40s %-20s\n", "ID", "Name", "Global Memory");
    printf("%-5s %-40s %-20s\n", "--", "----", "--------------");

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: Unable to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        // Convert global memory to gigabytes for easier reading
        double gbytes = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        printf("%-5d %-40s %-20.2f GB\n", dev, prop.name, gbytes);
    }

    return 0;
}
```