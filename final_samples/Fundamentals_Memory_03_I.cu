/*
Perform a three-way copy: HtoD to `d_a`, DtoD from `d_a` to `d_b`, DtoD from `d_b` to `d_c`. Verify `d_c`.

Thinking process:
- The goal is to demonstrate data transfer between host and device, and device-to-device copy operations.
- Allocate a small array on the host, initialize it with known values.
- Allocate three device arrays: d_a, d_b, d_c.
- Use cudaMemcpy to copy data from host to d_a (HtoD).
- Use cudaMemcpy again to copy from d_a to d_b (DtoD).
- Use cudaMemcpy again to copy from d_b to d_c (DtoD).
- Copy d_c back to host (DtoH) and verify that each element matches the original host data.
- Implement error checking after each CUDA API call to catch and report failures.
- Free all allocated memory at the end.
- Keep the code simple and self-contained, suitable for compiling with nvcc as a .cu file.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %d: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const size_t N = 10;
    const size_t bytes = N * sizeof(int);

    // Host allocation
    int *h_a = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    if (!h_a || !h_c) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<int>(i);
    }

    // Device allocations
    int *d_a = nullptr;
    int *d_b = nullptr;
    int *d_c = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    // HtoD: copy h_a to d_a
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // DtoD: copy d_a to d_b
    CHECK_CUDA(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));

    // DtoD: copy d_b to d_c
    CHECK_CUDA(cudaMemcpy(d_c, d_b, bytes, cudaMemcpyDeviceToDevice));

    // DtoH: copy d_c back to h_c
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verification
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_a[i] != h_c[i]) {
            fprintf(stderr, "Mismatch at index %zu: host %d vs device %d\n",
                    i, h_a[i], h_c[i]);
            ok = false;
            break;
        }
    }

    if (ok) {
        printf("Verification passed: d_c matches original data.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Cleanup
    free(h_a);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
