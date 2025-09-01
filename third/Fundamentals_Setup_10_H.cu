/*
Query and print the GPU's clock rate in GHz.

Thinking:
- Use the CUDA runtime API to access device properties.
- The property `clockRate` in `cudaDeviceProp` is expressed in kHz.
- To convert kHz to GHz, divide by 1,000,000 (1e6).
- We'll query the default device (device 0) and print its clock rate in GHz.
- Include basic error handling for CUDA API calls.
- The program will compile with `nvcc` and run on any system with a CUDA-capable GPU.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

int main() {
    int device = 0;  // Query the first CUDA device

    // Retrieve device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // Convert clock rate from kHz to GHz
    double clockGHz = static_cast<double>(prop.clockRate) / 1e6;

    // Print the result
    printf("GPU Clock Rate: %.3f GHz\n", clockGHz);

    return 0;
}
