/*
To verify the DtoD copy, copy d_b back to a host array and check its contents.

Thinking:
- Allocate two arrays on the host (h_a and h_b) and initialize h_a with a predictable pattern.
- Allocate corresponding device arrays (d_a and d_b).
- Copy the contents of h_a to d_a using cudaMemcpy.
- Perform a device-to-device copy from d_a to d_b using cudaMemcpy with the cudaMemcpyDeviceToDevice flag.
- Copy the contents of d_b back to h_b using cudaMemcpy.
- Compare h_b to the original pattern in h_a (or simply compare to the expected values) to verify that the data remained intact through the DtoD copy.
- Include error checking after each CUDA call and print a success or failure message.
- Use a small array size (e.g., 10 elements) for simplicity.
*/

#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

int main(void) {
    // Host arrays
    int h_a[ARRAY_SIZE];
    int h_b[ARRAY_SIZE];

    // Initialize h_a with known values
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_a[i] = i * 2;          // example pattern: 0, 2, 4, ...
    }

    // Device pointers
    int *d_a = NULL;
    int *d_b = NULL;

    size_t bytes = ARRAY_SIZE * sizeof(int);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));

    // Copy h_a to d_a
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Device-to-device copy from d_a to d_b
    CUDA_CHECK(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    // Copy d_b back to h_b
    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    // Verify that h_b matches the original pattern
    int success = 1;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        if (h_b[i] != h_a[i]) {
            fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n",
                    i, h_a[i], h_b[i]);
            success = 0;
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

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
