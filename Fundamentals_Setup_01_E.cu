```cuda
/*
 * Aim of the program (exact wording from the prompt):
 * Query and print the memory clock rate of the primary CUDA device in kilohertz (kHz).
 *
 * Thought process:
 * 1. To obtain the memory clock rate, we need to use the CUDA Runtime API.
 * 2. The memory clock rate is stored in the cudaDeviceProp structure returned
 *    by cudaGetDeviceProperties(). The field `memoryClockRate` gives the rate in
 *    kHz. The prompt explicitly asks to print it in kHz, so we can simply print
 *    that value.
 * 3. We should query the primary device using cudaGetDevice(). That gives us
 *    the current device index.
 * 4. Error checking: after each CUDA call we should check the returned
 *    cudaError_t and, if it indicates failure, print an error message and exit.
 * 5. The program is very small, so we can keep it all in main(). We use
 *    stdio.h for printing.
 * 6. Since we are writing a .cu file, we need to include <cuda_runtime.h>.
 * 7. Finally, we return 0 on success.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                  \
        }                                                          \
    } while (0)

int main(void) {
    int device = 0;
    // Get the current device (primary device)
    CHECK_CUDA(cudaGetDevice(&device));

    // Retrieve properties of the device
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // Print the memory clock rate in kHz
    printf("Memory Clock Rate of device %d (%s): %d kHz\n",
           device, prop.name, prop.memoryClockRate);

    return EXIT_SUCCESS;
}
```