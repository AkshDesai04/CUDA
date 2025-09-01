/*
 * Aim of the program: Query and print the number of multiprocessors on the primary CUDA device (device 0).
 *
 * Thinking process:
 * 1. The goal is to query the number of multiprocessors on the default primary device (device 0).
 * 2. In CUDA C, the device properties can be retrieved using cudaGetDeviceProperties().
 * 3. The device number can be specified as 0. Optionally we can call cudaSetDevice(0) to ensure we are on that device.
 * 4. The cudaDeviceProp struct contains a field called multiProcessorCount which holds the desired value.
 * 5. We'll print the value using printf. Basic error handling will be included to catch CUDA API failures.
 * 6. The program will be a minimal self-contained .cu file that can be compiled with nvcc.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaError_t err;

    // Optionally set the device to ensure we are on device 0
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error setting device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    printf("Number of multiprocessors on device %d: %d\n", device, prop.multiProcessorCount);

    return 0;
}
