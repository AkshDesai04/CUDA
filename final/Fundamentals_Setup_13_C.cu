/*
Aim: Iterate through all devices and print the clock rate for each one.
Thinking: The program uses the CUDA runtime API to query the number of CUDA-capable devices, then loops over each device index, retrieves its properties via cudaGetDeviceProperties, and prints the clockRate field. The code includes error handling for the API calls and prints informative messages to stderr if any call fails. The output includes the total number of devices and for each device, its name and clock rate in kHz. This program should be compiled with nvcc and run on a system with the CUDA toolkit installed.
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

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s, Clock rate: %d kHz\n", i, prop.name, prop.clockRate);
    }

    return 0;
}
