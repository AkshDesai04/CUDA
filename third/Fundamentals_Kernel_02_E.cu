```cuda
/*
Pass an integer as an argument to the kernel and print its value from the device.

Thinking:
1. The requirement is to pass an integer from host to device kernel and
   print it on the device side using device-side printf.
2. Device-side printf is supported in CUDA since early releases, but
   the target GPU architecture must support it (compute capability >= 2.0).
3. The host program will:
   a. Parse an integer from command line arguments (argv[1]).
   b. Launch a simple kernel with a single block and a single thread.
   c. The kernel will receive the integer and call `printf`.
   d. After the kernel launch, the host will synchronize with the device.
4. Error handling:
   - Use a simple macro to check CUDA runtime API calls.
   - Handle cases where the user does not provide an argument or provides
     an invalid integer.
5. No external libraries or complex logic are needed; keep it minimal and
   illustrative of the concept.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple macro to check CUDA errors
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),         \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that prints the integer value passed from host
__global__ void printKernel(int val) {
    // Device-side printf requires printf support in the GPU
    printf("Device: received value = %d\n", val);
}

int main(int argc, char* argv[]) {
    // Ensure user provided an integer argument
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <integer>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Convert argument to integer
    char* endptr;
    long long_val = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0') {
        fprintf(stderr, "Invalid integer: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    int host_val = (int)long_val;

    // Launch kernel with one block and one thread
    printKernel<<<1, 1>>>(host_val);
    CUDA_CHECK(cudaGetLastError());          // Check for launch errors

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Host: kernel execution completed.\n");

    return EXIT_SUCCESS;
}
```