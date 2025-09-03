/*
 * The aim of the program is to verify the DtoD copy, copy d_b back to a host array and check its contents.
 *
 * Thinking:
 * 1. Allocate two device arrays: d_a and d_b. d_a will be initialized from the host and then copied to d_b
 *    using a device-to-device copy (cudaMemcpyDeviceToDevice). This demonstrates a simple DtoD transfer.
 *
 * 2. To verify that the copy was successful, we copy the contents of d_b back to a host array (h_b_check)
 *    using cudaMemcpyDeviceToHost. We then compare h_b_check element‑by‑element with the original host
 *    array (h_a) that we copied into d_a. If all elements match, the DtoD copy worked correctly.
 *
 * 3. The program is intentionally minimal: it uses an integer array of small size for clarity.
 *    Error checking is performed after each CUDA call to ensure robustness.
 *
 * 4. The program flow:
 *    - Allocate host arrays h_a, h_b_check.
 *    - Allocate device arrays d_a, d_b.
 *    - Initialize h_a with distinct values.
 *    - Copy h_a to d_a (Host → Device).
 *    - Copy d_a to d_b (Device → Device).
 *    - Copy d_b back to h_b_check (Device → Host).
 *    - Compare h_a and h_b_check; report success or mismatch.
 *    - Clean up memory on both host and device.
 *
 * 5. Compile with:
 *      nvcc -o dtoD_check dtoD_check.cu
 *
 * 6. Run:
 *      ./dtoD_check
 *
 * 7. Expected output:
 *      All values match. DtoD copy verified successfully.
 *      or
 *      Mismatch detected at index i: expected X, got Y
 *
 * This code demonstrates the essential steps for copying data between device memory locations and back
 * to the host for verification.
 */

#include <cstdio>
#include <cstdlib>

#define N 10  // Size of the arrays

int main() {
    // Host arrays
    int *h_a = nullptr;
    int *h_b_check = nullptr;

    // Device arrays
    int *d_a = nullptr;
    int *d_b = nullptr;

    // Allocate host memory
    h_a = (int*)malloc(N * sizeof(int));
    h_b_check = (int*)malloc(N * sizeof(int));
    if (!h_a || !h_b_check) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array h_a with distinct values
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 10;  // e.g., 0, 10, 20, ...
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_a, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_a failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**)&d_b, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_b failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        return EXIT_FAILURE;
    }

    // Copy h_a to d_a (Host -> Device)
    err = cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HtoD failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return EXIT_FAILURE;
    }

    // Perform DtoD copy from d_a to d_b
    err = cudaMemcpy(d_b, d_a, N * sizeof(int), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DtoD failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return EXIT_FAILURE;
    }

    // Copy d_b back to host array h_b_check (Device -> Host)
    err = cudaMemcpy(h_b_check, d_b, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        cudaFree(d_b);
        return EXIT_FAILURE;
    }

    // Verify that h_b_check matches original h_a
    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (h_a[i] != h_b_check[i]) {
            fprintf(stderr, "Mismatch detected at index %d: expected %d, got %d\n",
                    i, h_a[i], h_b_check[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("All values match. DtoD copy verified successfully.\n");
    } else {
        printf("DtoD copy verification failed.\n");
    }

    // Clean up
    free(h_a);
    free(h_b_check);
    cudaFree(d_a);
    cudaFree(d_b);

    return match ? EXIT_SUCCESS : EXIT_FAILURE;
}
