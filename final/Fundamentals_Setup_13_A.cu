/*
 * Aim of the program: Convert and print the clock rate in megahertz (MHz).
 *
 * Thought process:
 * 1. In CUDA, each device has a property called clockRate, which is expressed in kilohertz (kHz).
 * 2. To obtain this property, we use the CUDA Runtime API: cudaGetDeviceProperties().
 * 3. The program will query the default device (device 0) for simplicity.
 * 4. The clockRate value (in kHz) will be converted to megahertz by dividing by 1000.0.
 * 5. The result will be printed in a human‑readable format, showing the device number,
 *    the device name, and the clock rate in MHz with three decimal places.
 * 6. Error checking is included to handle any CUDA API failures gracefully.
 *
 * The program is written in plain C using the CUDA Runtime API. It compiles as a .cu file
 * and can be built with: nvcc -o clock_rate clock_rate.cu
 *
 * This code is self‑contained and does not rely on any external headers beyond the
 * standard CUDA headers and the standard C library.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s (in %s line %d)\n",             \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            return 1;                                                      \
        }                                                                   \
    } while (0)

int main(void) {
    int device = 0;
    cudaDeviceProp prop;

    /* Get the current device (default is 0) */
    CHECK_CUDA(cudaGetDevice(&device));

    /* Retrieve properties of the device */
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    /* Convert clockRate from kHz to MHz */
    double clockMHz = prop.clockRate / 1000.0;

    /* Print the result */
    printf("Device %d (%s) Clock rate: %.3f MHz\n",
           device, prop.name, clockMHz);

    return 0;
}
