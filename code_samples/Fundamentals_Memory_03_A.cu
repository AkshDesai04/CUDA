```cpp
/*
 * Aim of the program:
 * To verify the DtoD copy, copy d_b back to a host array and check its contents.
 *
 * Thinking:
 * 1. The program demonstrates a simple device-to-device (DtoD) memory copy in CUDA.
 * 2. It starts by creating a host array `h_a` with known values.
 * 3. Device memory `d_a` and `d_b` are allocated.
 * 4. The host data is copied to `d_a` using `cudaMemcpyHostToDevice`.
 * 5. A DtoD copy is performed: `cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice)`.
 * 6. The data from `d_b` is copied back to the host array `h_b` using `cudaMemcpyDeviceToHost`.
 * 7. Finally, the program compares `h_a` and `h_b` to confirm that the DtoD copy was successful.
 * 8. Basic error checking is included after each CUDA API call.
 * 9. The program prints a success message if all values match, otherwise it reports mismatches.
 * 
 * This code is a minimal, self-contained example that can be compiled with nvcc:
 *   nvcc -o dtd_copy_example dtd_copy_example.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void) {
    const size_t N = 1024;               // Number of elements
    const size_t bytes = N * sizeof(int);

    // Host arrays
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array h_a with known values
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = (int)i;
    }

    // Device pointers
    int *d_a = nullptr;
    int *d_b = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));

    // Copy host data to device (h_a -> d_a)
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Perform device-to-device copy: d_a -> d_b
    CUDA_CHECK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    // Copy data back from d_b to host array h_b
    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    // Verify that h_b matches h_a
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_a[i] != h_b[i]) {
            fprintf(stderr, "Mismatch at index %zu: h_a=%d, h_b=%d\n",
                    i, h_a[i], h_b[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("DtoD copy verified successfully. All values match.\n");
    } else {
        printf("DtoD copy verification failed.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a);
    free(h_b);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```