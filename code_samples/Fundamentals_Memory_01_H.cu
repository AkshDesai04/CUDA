```cpp
/*
Aim of the program:
Allocate and copy a much larger array, e.g., 1 million integers.

Thought Process:
The goal is to demonstrate allocation of a large array on both host (CPU) and device (GPU),
and to perform a memory copy from host to device and optionally back for verification.
1. Define the size of the array: 1 million integers (1'000'000).
2. Allocate host memory using malloc (or new) and initialize it with known values.
3. Allocate device memory using cudaMalloc.
4. Use cudaMemcpy to transfer the data from host to device.
5. Optionally copy the data back from device to a second host buffer to verify the transfer.
6. Free all allocated memory and check for CUDA errors after each CUDA API call.
7. Provide basic timing (optional) to illustrate the cost of the transfer.

The program uses standard CUDA runtime API calls and prints simple status messages.
It is intentionally kept minimal to focus on the allocation and copy operations.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000  // 1 million integers

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void)
{
    // Allocate host memory
    int *h_data = (int *)malloc(N * sizeof(int));
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;  // simple pattern
    }

    // Allocate device memory
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, N * sizeof(int)));

    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    printf("Successfully copied %d integers from host to device.\n", N);

    // Optional: copy back to verify
    int *h_verify = (int *)malloc(N * sizeof(int));
    if (h_verify == NULL) {
        fprintf(stderr, "Failed to allocate host verify memory\n");
        cudaFree(d_data);
        free(h_data);
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaMemcpy(h_verify, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Successfully copied %d integers from device back to host.\n", N);

    // Verify data integrity
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (h_verify[i] != h_data[i]) {
            mismatches++;
            if (mismatches <= 10) {  // report first few mismatches
                fprintf(stderr, "Mismatch at index %d: host=%d, device=%d\n",
                        i, h_data[i], h_verify[i]);
            }
        }
    }
    if (mismatches == 0) {
        printf("Verification passed: all data matches.\n");
    } else {
        printf("Verification failed: %d mismatches found.\n", mismatches);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
    free(h_verify);

    return EXIT_SUCCESS;
}
```