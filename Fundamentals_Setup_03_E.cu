/*
Aim: Query and print the maximum memory pitch in bytes allowed for memory allocations.

Thinking:
To obtain the maximum memory pitch, we need to query the device properties via the CUDA Runtime API. 
The struct `cudaDeviceProp` contains a field `maxPitch` which represents the maximum pitch in bytes 
that can be used for pitched memory allocations on the device.  
The program will:

1. Determine the current device (or use device 0 by default).
2. Retrieve the device properties using `cudaGetDeviceProperties`.
3. Print the value of `maxPitch`.
4. Include basic error checking to ensure that CUDA calls succeed.

The program is written in standard C (with CUDA extensions) and compiled with `nvcc` to produce a `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro to check CUDA runtime API return status */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

int main(void) {
    int device = 0;  // Default to device 0
    cudaDeviceProp prop;

    /* If we want to query the current device, uncomment the following line:
    CUDA_CHECK(cudaGetDevice(&device));
    */

    /* Retrieve device properties */
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    /* Print the maximum pitch in bytes */
    printf("Device %d: Maximum memory pitch = %zu bytes\n", device, prop.maxPitch);

    return EXIT_SUCCESS;
}
