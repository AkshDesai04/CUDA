/*
 * Aim of the program (word for word):
 * For each device, print its memory bus width in bits.
 *
 * Thinking and design:
 * 1. Use CUDA Runtime API to discover all CUDA capable devices.
 *    - Call cudaGetDeviceCount to get the number of devices.
 * 2. Iterate over each device index.
 *    - For each device, use cudaGetDeviceProperties to obtain a cudaDeviceProp struct.
 * 3. From the struct, retrieve the memoryBusWidth field, which gives the bus width in bits.
 * 4. Print the device index, name, and memory bus width in a readable format.
 * 5. Include basic error checking after each CUDA call.
 * 6. Keep the code minimal and portable: only standard C/C++ headers and CUDA runtime.
 *
 * The program is a simple console application that outputs one line per device.
 * Example output:
 *   Device 0: NVIDIA GeForce GTX 1080 Ti - Memory Bus Width: 352 bits
 *
 * Note: The code assumes that a CUDA-enabled GPU is present and that the
 * CUDA runtime library is linked during compilation.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: cudaGetDeviceProperties failed for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s - Memory Bus Width: %d bits\n",
               dev, prop.name, prop.memoryBusWidth);
    }

    return 0;
}
