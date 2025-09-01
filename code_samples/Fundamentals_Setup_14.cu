/*
Write a program to check if the primary GPU can overlap data copying with kernel execution. This capability is crucial for performance and is indicated by the asyncEngineCount property (a value > 0).

Thinking:
- The asyncEngineCount field in cudaDeviceProp indicates the number of asynchronous engines a GPU has; if >0, it can overlap copies with kernel execution.
- We query the current device properties via cudaGetDeviceProperties.
- After retrieving asyncEngineCount, we print whether overlapping is supported.
- Include error checking for CUDA API calls.
- Use a simple CUDA C program structure (.cu file) with main() function.
- No actual kernel launch is needed, but the program can be extended to test overlapping.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main(void)
{
    int device = 0; /* primary device */
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device %d: %s\n", device, prop.name);
    printf("asyncEngineCount: %d\n", prop.asyncEngineCount);

    if (prop.asyncEngineCount > 0) {
        printf("This GPU can overlap data copying with kernel execution.\n");
    } else {
        printf("This GPU cannot overlap data copying with kernel execution.\n");
    }

    return EXIT_SUCCESS;
}
