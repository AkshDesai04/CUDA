/*
Allocate and copy a much larger array, e.g., 1 million integers.
To meet the user's request I first need to interpret what “allocate and copy a much larger array” means in a CUDA context. The simplest interpretation is: allocate a large host array, allocate a corresponding device array, and copy the data from host to device. This demonstrates basic CUDA memory operations and ensures that we are handling a sizable amount of data (1 million integers ≈ 4 MB), which is a reasonable size to illustrate memory allocation and transfer without overwhelming typical GPU memory limits.

My plan for the program is:
1. Define the number of elements (1,000,000) and compute the required byte size.
2. Allocate a host array of ints and initialize it with some pattern (e.g., the index value).
3. Allocate a device array of the same size using cudaMalloc.
4. Copy the data from the host to the device using cudaMemcpy with cudaMemcpyHostToDevice.
5. Perform a small verification step: copy the data back to a second host array and confirm that the contents match the original.
6. Clean up all allocated memory and report success or any errors.
7. Include proper CUDA error checking macros to simplify debugging.

The code is self‑contained and can be compiled with `nvcc` (e.g., `nvcc -o copy_large_array copy_large_array.cu`). It prints messages indicating progress and whether the copy succeeded.

Here is the complete .cu file:
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            return EXIT_FAILURE;                                                \
        }                                                                        \
    } while (0)

int main(void) {
    const size_t N = 1000000;          // 1 million integers
    const size_t size_bytes = N * sizeof(int);

    printf("Allocating host array of %zu integers (%zu bytes)...\n", N, size_bytes);
    int *h_array = (int*)malloc(size_bytes);
    if (h_array == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with some pattern
    for (size_t i = 0; i < N; ++i) {
        h_array[i] = (int)i;
    }

    // Allocate device memory
    printf("Allocating device array of %zu integers (%zu bytes)...\n", N, size_bytes);
    int *d_array = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_array, size_bytes));

    // Copy data from host to device
    printf("Copying data from host to device...\n");
    CUDA_CHECK(cudaMemcpy(d_array, h_array, size_bytes, cudaMemcpyHostToDevice));

    // Verification: copy back to a second host array
    printf("Copying data back from device to host for verification...\n");
    int *h_array_copy = (int*)malloc(size_bytes);
    if (h_array_copy == NULL) {
        fprintf(stderr, "Failed to allocate host copy memory.\n");
        cudaFree(d_array);
        free(h_array);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_array_copy, d_array, size_bytes, cudaMemcpyDeviceToHost));

    // Simple check
    bool match = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_array[i] != h_array_copy[i]) {
            fprintf(stderr, "Mismatch at index %zu: original %d, copy %d\n",
                    i, h_array[i], h_array_copy[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("Verification succeeded: all data matches.\n");
    } else {
        printf("Verification failed: data mismatch detected.\n");
    }

    // Clean up
    printf("Cleaning up resources...\n");
    CUDA_CHECK(cudaFree(d_array));
    free(h_array);
    free(h_array_copy);

    printf("Program completed successfully.\n");
    return EXIT_SUCCESS;
}
