/*
Aim of the program:
Write a program that checks if a hypothetical block configuration of (32, 32, 2) would be valid.

Thought process:
- The CUDA runtime provides the device properties via cudaGetDeviceProperties. 
- For a block configuration to be valid, each dimension (x, y, z) must be less than or equal to the device's maxThreadsDim[0..2].
- Additionally, the total number of threads in a block (x * y * z) must not exceed maxThreadsPerBlock.
- The program will query the properties of the current device (device 0 by default), compare them with the hard‑coded block configuration (32, 32, 2), and report whether the configuration is valid or not.
- Error handling for CUDA API calls will be performed using a simple macro.
- The program prints the block configuration, the device limits, and the result.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            return 1;                                                   \
        }                                                                \
    } while (0)

int main(void)
{
    /* Hard‑coded block dimensions to test */
    const int blockDimX = 32;
    const int blockDimY = 32;
    const int blockDimZ = 2;

    /* Get device properties for device 0 */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Device 0 properties:\n");
    printf("  maxThreadsDim[0]: %d\n", prop.maxThreadsDim[0]);
    printf("  maxThreadsDim[1]: %d\n", prop.maxThreadsDim[1]);
    printf("  maxThreadsDim[2]: %d\n", prop.maxThreadsDim[2]);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("\n");

    /* Check individual dimensions */
    bool dimValid = true;
    if (blockDimX > prop.maxThreadsDim[0]) {
        printf("Error: blockDim.x (%d) exceeds maxThreadsDim[0] (%d)\n",
               blockDimX, prop.maxThreadsDim[0]);
        dimValid = false;
    }
    if (blockDimY > prop.maxThreadsDim[1]) {
        printf("Error: blockDim.y (%d) exceeds maxThreadsDim[1] (%d)\n",
               blockDimY, prop.maxThreadsDim[1]);
        dimValid = false;
    }
    if (blockDimZ > prop.maxThreadsDim[2]) {
        printf("Error: blockDim.z (%d) exceeds maxThreadsDim[2] (%d)\n",
               blockDimZ, prop.maxThreadsDim[2]);
        dimValid = false;
    }

    /* Check total threads per block */
    int totalThreads = blockDimX * blockDimY * blockDimZ;
    bool totalValid = true;
    if (totalThreads > prop.maxThreadsPerBlock) {
        printf("Error: total threads per block (%d) exceeds maxThreadsPerBlock (%d)\n",
               totalThreads, prop.maxThreadsPerBlock);
        totalValid = false;
    }

    /* Final verdict */
    if (dimValid && totalValid) {
        printf("\nBlock configuration (%d, %d, %d) is valid for this device.\n",
               blockDimX, blockDimY, blockDimZ);
    } else {
        printf("\nBlock configuration (%d, %d, %d) is NOT valid for this device.\n",
               blockDimX, blockDimY, blockDimZ);
    }

    return 0;
}
