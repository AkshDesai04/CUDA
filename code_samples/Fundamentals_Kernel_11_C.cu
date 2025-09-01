```cpp
/*
Aim of the program:
In the host, allocate the struct using `cudaMallocManaged` instead of `cudaMalloc`. Initialize on host, launch kernel to modify, synchronize, and then access directly on host to verify.

Thinking:
1. Define a simple C++ struct (`MyStruct`) with a couple of fields.
2. Use `cudaMallocManaged` to allocate memory that is accessible from both host and device. This eliminates the need for explicit copies.
3. Initialize the struct fields on the host side, demonstrating that host code can set initial values directly.
4. Write a simple kernel (`modifyStruct`) that receives a pointer to `MyStruct` and modifies its fields (e.g., increment the integer and double the float). This kernel will be launched with a single thread because we only have one instance.
5. After launching the kernel, call `cudaDeviceSynchronize()` to wait for the device to finish, ensuring that the modifications are complete before the host reads them again.
6. Finally, print the struct contents on the host to verify that the kernel successfully updated the data.
7. Include basic error checking after CUDA API calls for robustness.
8. Compile with `nvcc` and run; the output should reflect the updated values.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Define a simple struct that will be shared between host and device
struct MyStruct {
    int   a;
    float b;
};

// Kernel that modifies the struct
__global__ void modifyStruct(MyStruct *ptr) {
    // Simple modifications: increment int and double the float
    ptr->a += 10;
    ptr->b *= 2.0f;
}

// Macro for checking CUDA errors
#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    __FUNCTION__, __FILE__, __LINE__,             \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    MyStruct *d_struct = NULL;

    // Allocate unified memory
    CHECK_CUDA(cudaMallocManaged(&d_struct, sizeof(MyStruct)));

    // Initialize on host
    d_struct->a = 42;
    d_struct->b = 3.14f;

    printf("Before kernel launch:\n");
    printf("  a = %d\n", d_struct->a);
    printf("  b = %f\n", d_struct->b);

    // Launch kernel with one block and one thread
    modifyStruct<<<1, 1>>>(d_struct);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Access directly on host after synchronization
    printf("\nAfter kernel execution:\n");
    printf("  a = %d\n", d_struct->a);
    printf("  b = %f\n", d_struct->b);

    // Clean up
    CHECK_CUDA(cudaFree(d_struct));

    return 0;
}
```