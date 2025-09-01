/*
Aim: From the host, use `cudaMemset` to zero out the struct's memory on the device after allocating it.

Thinking:
The task requires a CUDA C program that demonstrates how to allocate a struct on the GPU, then use the host API `cudaMemset` to set all bytes of that allocated memory to zero. To keep the example clear and self‑contained, we will:

1. Define a simple `MyStruct` type containing a few primitive fields (`int`, `float`, `double`).
2. Allocate device memory for one instance of `MyStruct` using `cudaMalloc`.
3. Immediately after allocation, invoke `cudaMemset(d_struct, 0, sizeof(MyStruct))` from the host.
4. Copy the zeroed struct back to the host with `cudaMemcpy` and print its contents to verify that all fields are indeed zero.
5. Include basic error checking after each CUDA call using a helper macro.
6. Wrap the logic in a `main` function and ensure proper cleanup with `cudaFree`.

The program will be written as a standard `.cu` file that can be compiled with `nvcc`. No additional kernels are required, as the focus is on using `cudaMemset` from the host to clear the memory of a device struct.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* Simple struct with different primitive types */
typedef struct {
    int   a;
    float b;
    double c;
} MyStruct;

int main(void) {
    MyStruct *d_struct = NULL;
    size_t structSize = sizeof(MyStruct);

    /* Allocate memory on the device */
    CUDA_CHECK(cudaMalloc((void**)&d_struct, structSize));
    printf("Allocated %zu bytes on device for MyStruct at %p\n",
           structSize, (void*)d_struct);

    /* Zero out the allocated memory using cudaMemset from host */
    CUDA_CHECK(cudaMemset(d_struct, 0, structSize));
    printf("Device memory zeroed using cudaMemset.\n");

    /* Copy back to host to verify */
    MyStruct h_struct;
    CUDA_CHECK(cudaMemcpy(&h_struct, d_struct, structSize, cudaMemcpyDeviceToHost));

    /* Print the contents of the struct */
    printf("Struct contents after memset:\n");
    printf("  a = %d\n", h_struct.a);
    printf("  b = %f\n", h_struct.b);
    printf("  c = %lf\n", h_struct.c);

    /* Clean up */
    CUDA_CHECK(cudaFree(d_struct));

    return 0;
}
