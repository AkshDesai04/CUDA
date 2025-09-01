/*
 * Aim: In the host, allocate the struct using cudaMallocManaged instead of cudaMalloc. Initialize on host, launch kernel to modify, synchronize, and then access directly on host to verify.
 *
 * Thinking:
 * 1. Define a struct MyStruct with a few members.
 * 2. Allocate memory for a MyStruct* on the device using cudaMallocManaged.
 * 3. Initialize the struct on the host before launching the kernel.
 * 4. Define a kernel that modifies the struct's fields (for example, increments or multiplies).
 * 5. After launching the kernel, synchronize the device with cudaDeviceSynchronize.
 * 6. On the host, after synchronization, access the struct and print the modified values to verify that the changes made in the kernel are visible on the host.
 * 7. Finally, free the managed memory.
 * This demonstrates unified memory and that host and device can share the same memory pointer.
 */

#include <cstdio>
#include <cuda_runtime.h>

struct MyStruct {
    int   a;
    float b;
};

__global__ void modifyStruct(MyStruct* s) {
    // Simple modification: increment integer and double the float
    s->a += 10;
    s->b *= 2.0f;
}

int main(void) {
    MyStruct* d_struct = nullptr;
    cudaError_t err;

    // Allocate unified memory for the struct
    err = cudaMallocManaged(&d_struct, sizeof(MyStruct));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Initialize the struct on the host
    d_struct->a = 5;
    d_struct->b = 3.14f;

    printf("Before kernel: a = %d, b = %f\n", d_struct->a, d_struct->b);

    // Launch kernel to modify the struct
    modifyStruct<<<1, 1>>>(d_struct);

    // Synchronize to ensure kernel has finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_struct);
        return 1;
    }

    // Access the struct on the host after synchronization
    printf("After kernel: a = %d, b = %f\n", d_struct->a, d_struct->b);

    // Clean up
    cudaFree(d_struct);

    return 0;
}
