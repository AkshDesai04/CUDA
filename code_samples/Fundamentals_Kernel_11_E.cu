```cuda
/*
 * Aim of the program: 
 * From the host, use `cudaMemset` to zero out the struct's memory on the device after allocating it.
 *
 * Thinking:
 * The user wants a CUDA program that demonstrates the usage of `cudaMemset` from the host to
 * clear a struct allocated on the device. To do this in a clear and minimal example we:
 * 1. Define a simple struct `MyStruct` with a few members of different types.
 * 2. Allocate memory for this struct on the device using `cudaMalloc`.
 * 3. Call `cudaMemset` from the host to set the allocated memory region to zero.
 * 4. Copy the struct back to the host with `cudaMemcpyDeviceToHost` so we can verify
 *    that all fields are indeed zeroed.
 * 5. Print the results from the host to confirm that the struct was zeroed.
 *
 * The program includes basic error checking after each CUDA runtime call using a helper
 * macro `CHECK_CUDA` to simplify the code and report any failures. This example can be
 * compiled with `nvcc` and run on any CUDA-capable device.
 */

#include <cstdio>
#include <cuda_runtime.h>

/* Simple struct with diverse member types */
struct MyStruct {
    int   a;
    float b;
    double c;
};

/* Helper macro for error checking */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    /* Allocate device memory for MyStruct */
    MyStruct *d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_ptr, sizeof(MyStruct)));

    /* Zero out the allocated memory on the device using cudaMemset */
    CHECK_CUDA(cudaMemset(d_ptr, 0, sizeof(MyStruct)));

    /* Copy the struct back to the host to verify it's zeroed */
    MyStruct h_struct;
    CHECK_CUDA(cudaMemcpy(&h_struct, d_ptr, sizeof(MyStruct), cudaMemcpyDeviceToHost));

    /* Print the values to confirm they are zero */
    printf("After cudaMemset:\n");
    printf("  a = %d\n", h_struct.a);
    printf("  b = %f\n", h_struct.b);
    printf("  c = %f\n", h_struct.c);

    /* Clean up device memory */
    CHECK_CUDA(cudaFree(d_ptr));

    return 0;
}
```