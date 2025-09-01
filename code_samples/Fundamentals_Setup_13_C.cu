```c
/*
Aim of the program:
Iterate through all devices and print the clock rate for each one.

Thinking:
To accomplish this, I will use the CUDA Runtime API. First, I need to include
the necessary headers: <stdio.h> for I/O and <cuda_runtime.h> for CUDA
runtime functions. I will then call cudaGetDeviceCount to find out how many
GPU devices are available. For each device index, I will retrieve a
cudaDeviceProp structure using cudaGetDeviceProperties. The clock rate is
stored in the 'clockRate' member of that structure, measured in kHz. I will
convert it to MHz for easier reading. Finally, I will print the device
index along with its clock rate. The program will be a simple C-style
main function that returns 0 on success. This code will be compiled with
nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Found %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        // Clock rate is in kHz; convert to MHz
        float clockMHz = prop.clockRate / 1000.0f;
        printf("Device %d: Clock Rate = %.2f MHz\n", dev, clockMHz);
    }

    return 0;
}
```