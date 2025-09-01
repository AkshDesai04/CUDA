/*
Query if the primary device has ECC (Error-Correcting Code) memory enabled and print the result.

Thinking process:
- We need to write a CUDA C program (.cu) that queries the ECC status of the primary GPU device.
- Use CUDA runtime API to get the current device ID via cudaGetDevice().
- Then use cudaGetDeviceProperties() to obtain a cudaDeviceProp structure.
- The cudaDeviceProp struct has an int field ECCEnabled that indicates whether ECC is enabled (1) or not (0).
- Print the status: "ECC memory is enabled" or "ECC memory is disabled".
- Include error checking: if cudaGetDevice() or cudaGetDeviceProperties() fails, print an error message.
- Use a helper macro CUDA_CHECK to wrap CUDA calls for concise error handling.
- The program is simple, so include only necessary headers: <stdio.h> and <cuda_runtime.h>.
- The main function will perform the steps above and return 0 on success.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(void) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (prop.ECCEnabled) {
        printf("ECC memory is enabled on device %d (%s).\n", device, prop.name);
    } else {
        printf("ECC memory is disabled on device %d (%s).\n", device, prop.name);
    }

    // Clean up CUDA context (optional, as program ends)
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
