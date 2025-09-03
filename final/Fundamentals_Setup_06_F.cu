/*
 * Aim of the program: Print the maximum z-dimension of a grid of blocks.
 *
 * Thinking process:
 * 1. In CUDA, the grid can be 3‑dimensional (x, y, z). The hardware imposes
 *    limits on each dimension which are accessible via cudaDeviceProp.maxGridSize.
 * 2. The maximum value for the z‑dimension is stored in maxGridSize[2] (index 2
 *    because the array is 0‑based: [x, y, z]).
 * 3. To retrieve this value we need to query the device properties using
 *    cudaGetDeviceProperties. This is done on the host.
 * 4. The program should set the device (device 0 is used for simplicity), call
 *    cudaGetDeviceProperties, and print the z‑dimension limit.
 * 5. A minimal error‑checking macro is added to simplify error handling.
 * 6. No kernel launch is necessary because the information comes from the
 *    device properties. However, a dummy kernel could be included for completeness.
 * 7. The code is self‑contained, uses standard headers (<stdio.h>, <cuda_runtime.h>),
 *    and can be compiled with nvcc.
 * 8. The program prints the result to stdout and exits with 0 on success.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple error checking macro */
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            return EXIT_FAILURE;                                 \
        }                                                         \
    } while (0)

/* Dummy kernel (not used but illustrates launching a 3D grid) */
__global__ void dummyKernel() {
    // No operation
}

int main(void) {
    /* Set device 0 */
    CUDA_CHECK(cudaSetDevice(0));

    /* Query device properties */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    /* Print the maximum z‑dimension of a grid of blocks */
    printf("Maximum grid z‑dimension: %d\n", prop.maxGridSize[2]);

    /* Optional: launch dummyKernel with max dimensions to ensure it runs */
    dim3 gridSize(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    dim3 blockSize(1);  /* minimal block size to avoid exceeding block limits */
    dummyKernel<<<gridSize, blockSize>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
