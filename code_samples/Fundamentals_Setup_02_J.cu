/*
For each device, print its maximum grid dimensions ([X, Y, Z]).

The goal of this program is to enumerate all CUDA-capable devices present in the system, query each device's
maximum grid dimension limits, and output those limits in a clear, readable format.  CUDA exposes these limits
via the cudaDeviceProp structure returned by cudaGetDeviceProperties, specifically the maxGridSize array.
The program will:

1. Use cudaGetDeviceCount() to determine how many CUDA devices are available.
2. For each device index:
   a. Call cudaSetDevice() to select the device (optional but good practice for multi-device contexts).
   b. Retrieve the device properties via cudaGetDeviceProperties().
   c. Extract the maxGridSize[0], maxGridSize[1], and maxGridSize[2] values, which represent the maximum
      number of blocks in the X, Y, and Z dimensions that a grid can have when launching kernels on that
      device.
   d. Print the device index and the three dimensions in the format requested.
3. Handle any CUDA runtime errors by printing a descriptive message and exiting gracefully.

The program uses the C standard I/O library for output and the CUDA Runtime API for device interrogation.
It is intentionally minimal, making it easy to compile with `nvcc` and run on any system with NVIDIA GPUs.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        // Select the device (optional but ensures subsequent calls refer to this device)
        err = cudaSetDevice(dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties(%d) failed: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: max grid dimensions = [%d, %d, %d]\n",
               dev,
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
    }

    return 0;
}
