/*
Aim: Deliberately introduce an error (e.g., pass a null pointer to `cudaMemcpy`) and see your macro in action.

Thinking:
The goal is to create a simple CUDA program that intentionally triggers a CUDA runtime error by passing a NULL device pointer to `cudaMemcpy`. We will implement a robust error-checking macro, `CUDA_CHECK`, that captures the error code, prints detailed information (file, line, error name, and error string), and exits the program. By running this program, we should see the macro in action, catching the `cudaErrorInvalidDevicePointer` error. The rest of the program is minimal: allocate host memory, populate it, attempt the faulty memcpy, and clean up. No additional functionality or kernel launches are required; the focus is on demonstrating error handling. The code is self-contained and ready to compile with `nvcc` as a `.cu` file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main(void) {
    const int N = 10;
    const size_t size = N * sizeof(int);
    int *h_a = NULL;
    int *d_a = NULL;  // Intentionally left NULL to cause error

    // Allocate and initialize host memory
    h_a = (int*)malloc(size);
    if (!h_a) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) h_a[i] = i;

    // Deliberately pass NULL device pointer to cudaMemcpy
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // This line will not be executed if error occurs
    printf("Data copied to device successfully.\n");

    // Clean up host memory
    free(h_a);
    return 0;
}
