/*
Write a program to get the ID of the currently active CUDA device using `cudaGetDevice` and print it.
The aim of this program is exactly as stated: we want to query which CUDA device is currently active (i.e., the one that will be used for subsequent CUDA operations unless changed) and output that device's numeric ID to the console. 

Thinking process:
1. Include the necessary headers: <stdio.h> for printing and <cuda_runtime.h> for the CUDA runtime API.
2. In the main function, declare an integer variable to hold the device ID.
3. Call `cudaGetDevice(&deviceId)` to retrieve the current active device index. This function returns a cudaError_t; we should check for success.
4. If the call is successful, print the device ID using `printf`. If it fails, print an error message using `cudaGetErrorString` to provide context.
5. Return 0 from main.

This is a minimal standalone CUDA C program that compiles with `nvcc` and prints the device ID. No device initialization or kernel launch is required; we're only querying state. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceId = -1;
    cudaError_t err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving current CUDA device: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    printf("Current active CUDA device ID: %d\n", deviceId);
    return 0;
}
