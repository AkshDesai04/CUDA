/*
aim: Write a program to get the ID of the currently active CUDA device using cudaGetDevice and print it.
thinking:
- Include the CUDA runtime header <cuda_runtime.h>.
- In the main function, declare an integer to hold the device ID.
- Call cudaGetDevice(&device). It returns a cudaError_t; we can check if the call succeeded.
- If the call is successful, print the device ID using printf.
- If the call fails, print an error message and return a non-zero exit code.
- The program is very small and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Current CUDA device ID: %d\n", device_id);
    return 0;
}
