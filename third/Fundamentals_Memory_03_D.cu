```cpp
/*
Aim of the program:
Use `cudaMemset` to initialize `d_a` to a value (e.g., 5) instead of copying from the host. Then perform the DtoD copy.

Thinking:
1. Allocate host memory for an array `h_a` and a destination array `h_b`.
2. Allocate two device buffers `d_a` and `d_b`.
3. Use `cudaMemset` to set all bytes of `d_a` to the byte value 5.  This will result in each 32‑bit integer in `d_a` containing the pattern 0x05050505, which is 84215045 in decimal.  (Note that `cudaMemset` works on bytes, not on whole ints.)
4. Perform a device‑to‑device copy from `d_a` to `d_b` using `cudaMemcpy` with the `cudaMemcpyDeviceToDevice` flag.
5. Copy `d_b` back to the host array `h_b` and print the values to verify that the copy worked.
6. Clean up all allocated memory and check for CUDA errors throughout.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

int main(void) {
    const int N = 10;                     // number of elements
    const size_t size = N * sizeof(int);  // total size in bytes

    // Allocate host memory
    int h_a[N];
    int h_b[N];

    // Allocate device memory
    int *d_a = NULL;
    int *d_b = NULL;
    CHECK(cudaMalloc((void**)&d_a, size));
    CHECK(cudaMalloc((void**)&d_b, size));

    // Initialize d_a with byte value 5
    CHECK(cudaMemset(d_a, 5, size));

    // Device to device copy from d_a to d_b
    CHECK(cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice));

    // Copy result back to host for verification
    CHECK(cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost));

    // Print the results
    printf("Contents of h_b after DtoD copy:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_b[%d] = %d\n", i, h_b[i]);
    }

    // Clean up
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    return 0;
}
```