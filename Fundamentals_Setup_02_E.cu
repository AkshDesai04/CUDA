/*
Write a program that finds the device with the most global memory and prints its name and index.

Thinking:
1. Include <cuda_runtime.h> and <stdio.h>.
2. Query number of CUDA devices with cudaGetDeviceCount.
3. Iterate over each device index, use cudaGetDeviceProperties to get the device's properties.
4. Keep track of the maximum global memory and the corresponding device index.
5. After the loop, print the device index and its name using printf.
6. Add basic error handling for CUDA API calls.
7. Use a macro for checking CUDA errors to keep the code clean.
8. Ensure the program compiles as a .cu file and can be built with nvcc.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* Macro to check CUDA errors */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",        \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),            \
                    cudaGetErrorString(err));                                 \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    int maxIndex = -1;
    size_t maxGlobalMem = 0;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        if (prop.totalGlobalMem > maxGlobalMem) {
            maxGlobalMem = prop.totalGlobalMem;
            maxIndex = i;
        }
    }

    if (maxIndex >= 0) {
        cudaDeviceProp maxProp;
        CUDA_CHECK(cudaGetDeviceProperties(&maxProp, maxIndex));
        printf("Device with most global memory:\n");
        printf("Index: %d\n", maxIndex);
        printf("Name : %s\n", maxProp.name);
        printf("Total Global Memory: %.2f GB\n",
               (double)maxProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    } else {
        printf("Could not determine device with most global memory.\n");
    }

    return EXIT_SUCCESS;
}
