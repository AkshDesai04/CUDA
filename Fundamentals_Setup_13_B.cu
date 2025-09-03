```cpp
/*
Aim of the program: Convert and print the clock rate in gigahertz (GHz).

Thinking process:
1. The CUDA runtime API provides the `cudaDeviceProp` structure which contains a member `clockRate`. This field gives the clock frequency of the GPU in kilohertz (kHz).
2. To convert this value to gigahertz (GHz), we need to divide by 1,000,000 (since 1 GHz = 1,000,000 kHz).
3. The program will:
   - Query the properties of the first CUDA device (device 0).
   - Extract the `clockRate` in kHz.
   - Convert it to GHz using a double precision calculation.
   - Print the result with a descriptive message.
4. Error handling is included to catch any CUDA API failures.
5. The code is written in CUDA C and can be compiled with `nvcc` into an executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            return EXIT_FAILURE;                                                 \
        }                                                                        \
    } while (0)

int main(void) {
    int device = 0; // Use device 0 by default
    cudaDeviceProp prop;

    // Get properties of the specified device
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // clockRate is given in kHz; convert to GHz
    double clockRateGHz = prop.clockRate * 1e-6; // kHz to GHz

    printf("Device %d (%s) clock rate: %.3f GHz\n",
           device, prop.name, clockRateGHz);

    return EXIT_SUCCESS;
}
```