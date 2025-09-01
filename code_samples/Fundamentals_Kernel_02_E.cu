/*
 * Pass an integer as an argument to the kernel and print its value from the device.
 *
 * This program demonstrates how to send a simple integer value from the host
 * to a CUDA kernel and have that kernel print the value using device-side
 * printf.  Device printf requires a GPU compute capability of at least
 * 2.0, so this code should be compiled with an architecture flag such as
 * -arch=sm_20 or higher.
 *
 * Design notes:
 * 1. The kernel receives an int parameter by value and immediately prints it.
 * 2. Host code optionally parses an integer from the command line; if none is
 *    supplied a default of 42 is used.
 * 3. After launching the kernel we call cudaDeviceSynchronize() to wait for
 *    completion and then check for any errors.
 * 4. Minimal error handling is performed via a simple macro to keep the code
 *    readable while still reporting CUDA API failures.
 * 5. No device memory is allocated because the integer is passed directly
 *    as a kernel argument.
 *
 * Usage example:
 *   nvcc -arch=sm_35 -o print_int print_int.cu
 *   ./print_int 123
 * This will print:
 *   Device: integer value = 123
 *
 * The program is intentionally concise and self-contained, illustrating the
 * basic pattern of passing scalar arguments to kernels and using device
 * printf for debugging or logging purposes.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple error-checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that prints the received integer
__global__ void print_int_kernel(int val) {
    printf("Device: integer value = %d\n", val);
}

int main(int argc, char *argv[]) {
    int host_val = 42; // default value

    if (argc > 1) {
        host_val = std::atoi(argv[1]);
    }

    // Launch kernel with one block and one thread
    print_int_kernel<<<1, 1>>>(host_val);

    // Wait for kernel to finish and check for errors
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
