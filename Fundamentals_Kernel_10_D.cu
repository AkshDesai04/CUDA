/*
Aim: Define a nested struct (a struct that contains another struct as a member). Pass it by value to a kernel and access a member of the inner struct.

Thinking:
- The task requires us to demonstrate the use of nested structs in a CUDA program.
- A nested struct is a struct that has another struct as one of its members. 
- We will create two structs: an inner struct named `Inner` that holds an integer `x`, and an outer struct named `Outer` that holds an `Inner` instance and another integer `id`.
- The kernel will accept an instance of `Outer` by value. When a struct is passed by value to a CUDA kernel, the compiler copies the struct into the kernel’s register/local memory, which is a typical, lightweight mechanism for small structs.
- Inside the kernel, we will access the inner struct’s member (`x`) and print both `x` and the outer `id` using `printf`. This demonstrates that the nested member is accessible correctly.
- The host code will create an `Outer` object, initialize its fields, and launch the kernel with a single thread (since we just need to demonstrate the data transfer). After the kernel launch, we will synchronize and check for errors.
- The program includes a small helper macro `CHECK_CUDA` to simplify error handling.
- The code is fully self-contained and can be compiled with `nvcc`. It uses only CUDA runtime API and standard C headers.

The final program will look like this: 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* Nested struct definitions */
struct Inner {
    int x;
};

struct Outer {
    Inner inner;  // Nested struct member
    int id;
};

/* Kernel that receives Outer by value and accesses inner.x */
__global__ void printNestedStruct(Outer o) {
    /* Use printf from device to output values */
    printf("Thread %d: Outer.id = %d, Inner.x = %d\n", threadIdx.x, o.id, o.inner.x);
}

int main(void) {
    /* Create and initialize Outer struct on host */
    Outer h_outer;
    h_outer.id = 7;
    h_outer.inner.x = 42;

    /* Launch kernel with one block and one thread */
    printNestedStruct<<<1, 1>>>(h_outer);

    /* Synchronize and check for kernel launch errors */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Clean up */
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
