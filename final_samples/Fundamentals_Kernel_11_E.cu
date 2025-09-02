/*
From the host, use `cudaMemset` to zero out the struct's memory on the device after allocating it.

Thinking:
1. Define a simple struct (`MyStruct`) with a few members (int, float, int).
2. Allocate memory for this struct on the GPU using `cudaMalloc`.
3. Immediately zero the entire memory region with `cudaMemset(d_struct, 0, sizeof(MyStruct))`.
   This should set all bytes of the struct to zero, effectively initializing all numeric fields to 0.
4. Copy the struct back to the host with `cudaMemcpy` to verify that the values are indeed zero.
5. Print the host-side values to demonstrate that the memset worked.
6. Add basic error checking after each CUDA call to aid debugging.
7. Compile with `nvcc` to produce a runnable executable.
*/

#include <cuda_runtime.h>
#include <stdio.h>

struct MyStruct {
    int   a;
    float b;
    int   c;
};

/* Helper macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    MyStruct *d_struct = NULL;
    size_t structSize = sizeof(MyStruct);

    /* Allocate device memory for the struct */
    CHECK_CUDA(cudaMalloc((void **)&d_struct, structSize));

    /* Zero out the struct's memory on the device */
    CHECK_CUDA(cudaMemset(d_struct, 0, structSize));

    /* Copy the struct back to the host to verify it was zeroed */
    MyStruct h_struct;
    CHECK_CUDA(cudaMemcpy(&h_struct, d_struct, structSize, cudaMemcpyDeviceToHost));

    /* Print the values to confirm they are zero */
    printf("After cudaMemset:\n");
    printf("  a = %d\n", h_struct.a);
    printf("  b = %f\n", h_struct.b);
    printf("  c = %d\n", h_struct.c);

    /* Clean up */
    CHECK_CUDA(cudaFree(d_struct));

    return 0;
}
