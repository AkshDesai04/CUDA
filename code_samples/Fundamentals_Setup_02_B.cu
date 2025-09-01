/*
Iterate through all devices and, for each one, print its multiprocessor count.

My approach:
1. Include the necessary headers: <stdio.h> for I/O and <cuda_runtime.h> for CUDA runtime functions.
2. Use cudaGetDeviceCount to determine how many CUDA-capable devices are present.
3. Iterate over each device index:
   - Retrieve device properties with cudaGetDeviceProperties.
   - Extract the multiprocessor count from the properties struct (multiProcessorCount).
   - Print the device index, name, and multiprocessor count.
4. Handle potential errors from CUDA API calls by checking the returned cudaError_t and printing an error message if necessary.
5. The program will run on the host; no device kernels are launched, so no GPU memory allocation or kernel execution is required.

The program is simple and demonstrates basic device enumeration in CUDA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
    }

    return 0;
}
