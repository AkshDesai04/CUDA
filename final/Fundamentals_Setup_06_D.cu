/*
Print the maximum x-dimension of a grid of blocks.

Thinking:
- The task is to obtain and display the maximum grid dimension along the X axis
  supported by the CUDA device.
- CUDA exposes this information in the `cudaDeviceProp` structure, specifically
  the `maxGridSize[0]` field.
- We query the properties of the current device using `cudaGetDeviceProperties`.
- No kernel launch is necessary; we can perform all operations on the host.
- For robustness, we wrap CUDA API calls in an error‑checking macro.
- The program will print the integer value to standard output.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__,                      \
                    cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("%d\n", prop.maxGridSize[0]);

    return 0;
}
