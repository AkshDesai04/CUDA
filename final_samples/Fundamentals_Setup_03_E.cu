/*
 * Query and print the maximum memory pitch in bytes allowed for memory allocations.
 *
 * ===================== Thought Process =====================
 *
 * 1. The goal is to find the maximum pitch (row alignment) that can be used with cudaMallocPitch.
 *    CUDA provides this information in the device properties structure `cudaDeviceProp`.
 *    The field is named `maxPitch` and is of type `size_t`.
 *
 * 2. Steps to achieve this:
 *    - Include the necessary CUDA runtime header (`cuda_runtime.h`) and standard I/O.
 *    - Optionally create a macro for error checking to make the code robust.
 *    - Query the number of CUDA-capable devices with `cudaGetDeviceCount`.
 *    - For device 0 (or any device), call `cudaGetDeviceProperties` to obtain the `cudaDeviceProp`.
 *    - Retrieve `prop.maxPitch` and print it in bytes.
 *
 * 3. Error handling:
 *    - Use a helper macro `CHECK_CUDA` that checks the return status of CUDA API calls.
 *    - If an error occurs, print an error message and exit.
 *
 * 4. Printing:
 *    - Use `printf` with `%zu` to correctly format `size_t` values.
 *
 * 5. The program is self‑contained and can be compiled with `nvcc`:
 *    `nvcc -o query_pitch query_pitch.cu`
 *
 * 6. The output will be something like:
 *    "Maximum memory pitch allowed for memory allocations: 2147483648 bytes"
 *
 * ===================== End of Thought Process =====================
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // Query the first device
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Maximum memory pitch allowed for memory allocations: %zu bytes\n",
           prop.maxPitch);

    return EXIT_SUCCESS;
}
