/*
Aim of the program: Write a program that attempts to set the device to an invalid index (e.g., 999) and use proper error checking to report that the device does not exist.

Thinking:
The task requires a CUDA C++ program that deliberately uses an invalid device index when calling cudaSetDevice. We must catch the error returned by cudaSetDevice and report it to the user in a meaningful way. To make the example clear, we will first query the number of available devices with cudaGetDeviceCount and print that out. Then we attempt to set an invalid device (999). Since cudaSetDevice returns a cudaError_t, we can check if it is not cudaSuccess. If an error occurs, we will print the error string using cudaGetErrorString. This will clearly show that the device does not exist. The program will be a simple single-file CUDA program with a main() function, including the necessary headers (<stdio.h> and <cuda_runtime.h>), and using standard I/O to display results. The program will compile with nvcc and run on any system with CUDA installed. The only output will be the error message indicating that the device index is invalid.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error querying device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Number of CUDA devices available: %d\n", deviceCount);

    // Attempt to set an invalid device index
    int invalidDeviceIndex = 999;
    err = cudaSetDevice(invalidDeviceIndex);
    if (err != cudaSuccess) {
        printf("Failed to set device %d: %s\n", invalidDeviceIndex, cudaGetErrorString(err));
    } else {
        printf("Successfully set device %d (this should not happen)\n", invalidDeviceIndex);
    }

    return 0;
}
