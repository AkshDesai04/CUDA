/*
Aim of the program: 
Query and print the maximum number of threads allowed in a single thread block for the primary CUDA device. This value, `maxThreadsPerBlock`, is a key constraint when designing kernels.

Thinking process:
1. We need to interact with the CUDA runtime API, so include <cuda_runtime.h>.
2. Use `cudaGetDeviceCount` to ensure there is at least one CUDA-capable device.
3. Select device 0 (primary) with `cudaSetDevice(0)` or just use `cudaGetDeviceProperties` with dev=0.
4. Retrieve device properties via `cudaGetDeviceProperties`, which fills a `cudaDeviceProp` struct.
5. Print `props.maxThreadsPerBlock` along with some context.
6. Handle any errors returned by CUDA API calls: if an error occurs, print a message and exit.
7. Keep the code minimal but clear, and include necessary headers.
8. Ensure the program is self-contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // Primary CUDA device
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Maximum threads per block for device %d (%s): %d\n",
           device, props.name, props.maxThreadsPerBlock);

    return EXIT_SUCCESS;
}
