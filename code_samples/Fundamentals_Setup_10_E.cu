/*
Query and print the maximum number of threads per multiprocessor.

The goal of this program is to report the hardware capability of each CUDA
device installed on the system, specifically the maximum number of threads
that can be resident on a single streaming multiprocessor (SM).  This
information is exposed by the CUDA Runtime API in the
`cudaDeviceProp.maxThreadsPerMultiProcessor` field.

The program follows these steps:
1. Use `cudaGetDeviceCount()` to determine how many CUDA-capable GPUs are
   present.  If no devices are found, the program exits gracefully.
2. For each device, call `cudaGetDeviceProperties()` to fill a
   `cudaDeviceProp` structure with all available properties.
3. Extract the `maxThreadsPerMultiProcessor` field from that structure.
4. Print the device index, name, and the queried maximum thread count.
5. Perform basic error checking on every CUDA Runtime API call and
   report any failures to `stderr`.

This approach ensures portability across CUDA-capable devices and
demonstrates the use of CUDA device queries from a plain C/C++ host
program.  The program is written in a single `.cu` file and can be
compiled with `nvcc`. 
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
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    }

    return 0;
}
