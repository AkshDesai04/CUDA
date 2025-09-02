/*
 * Aim: Use `sizeof()` on the host and device to print the size of the struct and verify they are the same.
 *
 * Thinking:
 * 1. Define a simple struct `MyStruct` that contains a few fields of different types.
 * 2. On the host, use the `sizeof` operator to obtain the size of `MyStruct` and print it.
 * 3. Create a CUDA kernel that also uses `sizeof(MyStruct)` and prints the result with `printf`.
 *    - Device printf requires a compute capability of at least 2.0, which is satisfied by most
 *      modern GPUs.  The kernel is trivial and just prints the size.
 * 4. In `main()`, launch the kernel, then synchronize and check for errors.
 * 5. Use a small helper macro for CUDA error checking to keep the code clean.
 * 6. The program is self‑contained in a single .cu file; it can be compiled with `nvcc` and run on
 *    a machine with an appropriate CUDA toolkit installed.
 *
 * The output will show the size computed on the host and the size printed from the device, and
 * they should match, confirming that the struct layout is the same on both sides.
 */

#include <cstdio>
#include <cuda_runtime.h>

/* Simple struct with various data types */
struct MyStruct {
    int    a;
    float  b;
    char   c[10];
};

/* CUDA error checking macro */
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Kernel that prints sizeof(MyStruct) on the device */
__global__ void printDeviceSize() {
    printf("Device sizeof(MyStruct) = %zu\n", sizeof(MyStruct));
}

int main() {
    /* Print sizeof on host */
    printf("Host sizeof(MyStruct) = %zu\n", sizeof(MyStruct));

    /* Launch kernel to print device size */
    printDeviceSize<<<1, 1>>>();
    CHECK_CUDA(cudaGetLastError());

    /* Wait for kernel to finish */
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}
