/*
Aim: Query and print the maximum number of threads allowed in a single thread block for the primary CUDA device. This value, `maxThreadsPerBlock`, is a key constraint when designing kernels.

Thinking: 
To accomplish this task, we need to interact with the CUDA Runtime API to retrieve device properties. The primary steps are:
1. Determine the number of CUDA-capable devices present in the system using `cudaGetDeviceCount`.
2. Select the primary device (device 0).
3. Obtain the device properties with `cudaGetDeviceProperties`, which fills a `cudaDeviceProp` structure containing many fields, including `maxThreadsPerBlock`.
4. Print the value of `maxThreadsPerBlock` to the console.
5. Handle possible error conditions, such as no devices being found or API call failures, by printing informative messages.
6. The program is written in C, compiled with `nvcc`, and produces a single .cu file.

The code below follows this plan, using standard C libraries for input/output and CUDA runtime functions for device queries. It includes minimal error checking to keep the example focused on the core requirement.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices detected.\n");
        return 0;
    }

    // Use the primary device (device 0)
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n",
                device, cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum threads per block for device %d (%s): %d\n",
           device, prop.name, prop.maxThreadsPerBlock);

    return 0;
}
