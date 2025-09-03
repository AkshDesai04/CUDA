/*
Query and print the maximum dimensions (x, y, z) of a thread block for the primary CUDA device. 
The result should be displayed clearly, for example: 'Max Block Dimensions: [X, Y, Z]'.

Thinking Process:
1. The CUDA runtime provides device properties via `cudaGetDeviceProperties`, which fills a `cudaDeviceProp` struct.
2. The struct contains `maxThreadsDim[3]` that represents the maximum number of threads per block in each dimension (x, y, z). 
3. We will query the properties of the current device (device 0, the primary device) and extract these three integers.
4. A simple error-checking macro will be used to ensure CUDA API calls succeed.
5. The program prints the dimensions in the required format and exits cleanly.
6. The code is written in standard CUDA C with the `.cu` extension, so it can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            return EXIT_FAILURE;                                    \
        }                                                           \
    } while (0)

int main(void) {
    int device = 0; // primary CUDA device
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int maxX = prop.maxThreadsDim[0];
    int maxY = prop.maxThreadsDim[1];
    int maxZ = prop.maxThreadsDim[2];

    printf("Max Block Dimensions: [%d, %d, %d]\n", maxX, maxY, maxZ);

    return EXIT_SUCCESS;
}
