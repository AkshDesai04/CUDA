/*
Aim: Query and print the maximum x-dimension of a thread block.

Thinking:
To obtain the maximum number of threads that can be launched in the x-dimension of a CUDA thread block,
we need to query the device properties via cudaGetDeviceProperties. The relevant field is
cudaDeviceProp::maxThreadsDim[0]. The program will:
1. Determine the number of available CUDA devices.
2. Select the first device (index 0) for simplicity.
3. Retrieve the device properties.
4. Extract and print the maxThreadsDim[0] value.
The code includes basic error checking for CUDA API calls and prints the result to stdout.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // Use the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Maximum x-dimension of a thread block on device %d: %d\n",
           device, prop.maxThreadsDim[0]);

    return EXIT_SUCCESS;
}
