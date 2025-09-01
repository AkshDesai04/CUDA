```cuda
/*
Aim of the program:
Define a simple C struct on the host. Pass an instance of this struct by value to a kernel. The kernel should print one of the struct's members to verify it was received correctly.

Thinking process:
1. Define a simple struct (e.g., MyStruct) that contains at least one integer and one float.
2. Create a kernel that takes this struct by value and uses printf to output one of its members.
3. In main, instantiate the struct, set its fields, and launch the kernel with a single thread/block.
4. Synchronize the device to ensure the printf output is flushed before the program exits.
5. Include minimal error checking for kernel launch.
6. The program should compile with nvcc and run on any CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct MyStruct {
    int   a;
    float b;
};

__global__ void printStruct(MyStruct s) {
    printf("Struct member a = %d\n", s.a);
}

int main(void) {
    MyStruct h_s;
    h_s.a = 42;
    h_s.b = 3.14f;

    // Launch kernel with one block and one thread
    printStruct<<<1, 1>>>(h_s);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to finish and flush printf buffer
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device synchronization error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
```