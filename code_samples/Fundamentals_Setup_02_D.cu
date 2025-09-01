/*
Iterate through all devices and print the maximum number of threads per block for each.

Thought process:
- To iterate over all CUDA-capable devices, use cudaGetDeviceCount to determine how many devices are present.
- For each device index from 0 to count-1:
    - Set the current device with cudaSetDevice.
    - Retrieve device properties using cudaGetDeviceProperties which fills a cudaDeviceProp struct.
    - From this struct, read the maxThreadsPerBlock field which indicates the maximum number of threads that can be launched in a single block on that device.
    - Print the device index and the corresponding maxThreadsPerBlock value.
- Add error checking after each CUDA API call to ensure any failure is reported and the program exits gracefully.
- If no devices are found, output a message indicating that no CUDA devices were detected.
- The program is written in C style, compiled with nvcc as a .cu file, and uses standard I/O for printing results.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main(void) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        // Set the device for subsequent operations
        CHECK_CUDA(cudaSetDevice(dev));

        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    }

    return EXIT_SUCCESS;
}
