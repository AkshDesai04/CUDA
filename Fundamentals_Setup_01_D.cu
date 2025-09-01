/*
Query and print the number of multiprocessors on the primary CUDA device (device 0).

The program is a minimal CUDA C example that retrieves the properties of the first CUDA
device (device 0) using `cudaGetDeviceProperties`. The relevant field for the number of
multiprocessors is `multiProcessorCount`. After querying the device properties, the program
prints this count to the standard output. Simple error handling is included to check
for CUDA API failures and to report them before exiting. No additional CUDA kernels or
memory operations are required for this query task. The program can be compiled with
`nvcc` and executed on a system that has at least one CUDA-capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            return EXIT_FAILURE;                                      \
        }                                                              \
    } while (0)

int main(void) {
    // Get properties of device 0
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Print the number of multiprocessors
    printf("Number of multiprocessors on device 0: %d\n", prop.multiProcessorCount);

    return EXIT_SUCCESS;
}
