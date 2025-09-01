```cpp
/* 
Aim of the program (as passed in the prompt, word for word):

Write a program that allocates two integer arrays, `d_a` and `d_b`, on the device. 
Initialize `d_a` by copying data from the host. Then, use `cudaMemcpy` with the 
`cudaMemcpyDeviceToDevice` kind to copy the contents of `d_a` directly to `d_b` 
without involving the host. 

Thinking:

1. We need a self‑contained CUDA C program, compiled with `nvcc` into a .cu file.
2. The program should allocate two integer arrays on the device (`d_a` and `d_b`).
3. The host array (`h_a`) must be initialized with some known values so we can verify
   that the device-to-device copy works.
4. We copy `h_a` → `d_a` using `cudaMemcpy` with kind `cudaMemcpyHostToDevice`.
5. Then we perform a device-to-device copy: `d_a` → `d_b` with kind 
   `cudaMemcpyDeviceToDevice`. No host involvement in this step.
6. To verify correctness, copy `d_b` back to a host array (`h_b`) and print it.
7. Add error checking after each CUDA call for robust diagnostics.
8. Use simple `int N = 10;` for array size, which is small enough to display easily.
9. All includes, definitions, and a `main()` function will be provided.
10. The entire code will be in one .cu file, with no extraneous text output, 
    satisfying the instruction to output only the file content. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Helper macro for error checking
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    // Host array initialization
    int h_a[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 10;  // Example values: 0, 10, 20, ...
    }

    // Allocate device arrays
    int *d_a = NULL;
    int *d_b = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_a, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, N * sizeof(int)));

    // Copy host to device (d_a)
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    // Device-to-device copy: d_a -> d_b
    CHECK_CUDA(cudaMemcpy(d_b, d_a, N * sizeof(int), cudaMemcpyDeviceToDevice));

    // Copy result back to host for verification
    int h_b[N];
    CHECK_CUDA(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Host array h_a: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    printf("Device-to-device copy result h_b: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}
```