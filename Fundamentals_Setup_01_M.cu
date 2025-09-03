/*
Query and print the major and minor compute capability numbers for the primary CUDA device separately.

Thought process:
- The primary CUDA device can be obtained via cudaGetDevice().
- Device properties are fetched with cudaGetDeviceProperties().
- The major and minor compute capability numbers are stored in the 'major' and 'minor' fields of the cudaDeviceProp structure.
- We need to handle potential CUDA API errors by checking the return status of each call.
- The output should display the major and minor numbers on separate lines to satisfy the requirement of printing them separately.
- The program is a simple console application written in C/C++ that compiles with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA API return values
#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            return 1;                                                     \
        }                                                                 \
    } while (0)

int main(void) {
    int device = 0; // Primary CUDA device
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // Print major and minor compute capability numbers separately
    printf("Major compute capability: %d\n", prop.major);
    printf("Minor compute capability: %d\n", prop.minor);

    return 0;
}
