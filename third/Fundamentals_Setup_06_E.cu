/*
 * Aim of the program: Print the maximum y-dimension of a grid of blocks.
 *
 * Thought process:
 * 1. In CUDA, the maximum grid dimensions are provided by the device properties
 *    accessible at runtime via `cudaGetDeviceProperties`. The member `maxGridSize`
 *    is an array where index 0 is the maximum x-dimension, index 1 is the maximum
 *    y-dimension, and index 2 is the maximum z-dimension.
 *
 * 2. To obtain these values, we need to:
 *    - Include the CUDA runtime header `<cuda_runtime.h>`.
 *    - Optionally, include `<stdio.h>` for printing.
 *    - Use `cudaGetDeviceCount` to ensure at least one CUDA device is available.
 *    - Retrieve the device properties for the first device (device 0) using
 *      `cudaGetDeviceProperties`.
 *    - Print `prop.maxGridSize[1]`, which corresponds to the maximum y-dimension.
 *
 * 3. For robustness, wrap CUDA API calls in a helper macro that checks the return
 *    value and reports an error if something goes wrong.
 *
 * 4. The program prints the device name and the maximum y-dimension value.
 *
 * 5. The program should compile to a .cu file and run on any system with CUDA
 *    installed, printing the desired value.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro to check CUDA errors */
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline void check(cudaError_t err, const char* func, const char* file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %d (%s) at %s:%d in %s\n",
                err, cudaGetErrorString(err), file, line, func);
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    /* Get the number of CUDA-capable devices */
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    /* Use the first device (device 0) */
    int device = 0;
    checkCudaErrors(cudaSetDevice(device));

    /* Retrieve device properties */
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    /* Print the maximum y-dimension of the grid of blocks */
    printf("Device %d: %s\n", device, prop.name);
    printf("Maximum y-dimension of a grid of blocks: %d\n", prop.maxGridSize[1]);

    return EXIT_SUCCESS;
}
