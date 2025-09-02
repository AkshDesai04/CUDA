```cuda
/*
For each device, print its maximum grid dimensions ([X, Y, Z]).

This program queries the number of CUDA-capable devices present in the system and then retrieves the properties of each device using cudaGetDeviceProperties. Each device's maximum grid dimensions are stored in the struct members dim3 maxGridSize, which hold the maximum allowed values for the grid dimensions along the X, Y, and Z axes. The program iterates over all devices, prints the device index, name, and the maximum grid dimensions in a readable format. If the query fails for any device, an error message is printed.

The key steps are:
1. Call cudaGetDeviceCount to determine the number of devices.
2. For each device index from 0 to count-1:
   - Use cudaGetDeviceProperties to fill a cudaDeviceProp struct.
   - Extract prop.maxGridSize.x, prop.maxGridSize.y, prop.maxGridSize.z.
   - Print these values along with the device name.
3. Handle any potential errors from CUDA API calls.

No kernel launches or GPU computations are performed; the program only retrieves device metadata and prints it. The code is written in C/C++ with CUDA extensions and can be compiled with nvcc as follows:
   nvcc -o max_grid_dimensions max_grid_dimensions.cu
Running the resulting executable will display the maximum grid dimensions for each CUDA-capable device on the system.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Max Grid Dimensions: [%d, %d, %d]\n\n",
               prop.maxGridSize.x,
               prop.maxGridSize.y,
               prop.maxGridSize.z);
    }

    return 0;
}
```