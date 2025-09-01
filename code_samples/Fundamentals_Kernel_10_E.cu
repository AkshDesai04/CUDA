```cpp
/*
Aim: Define a struct that contains a small, fixed-size array (e.g., `float vec[3]`). Pass it by value and have the kernel print the array elements.

Thinking:
- We need a simple CUDA program that defines a struct with a small fixed-size array.
- The struct will be `struct MyStruct { float vec[3]; };` which is trivially copyable.
- In CUDA, kernel arguments are copied from host to device automatically, so passing the struct by value to the kernel is straightforward.
- The kernel will simply print the three elements of the array using `printf`. CUDA supports device-side printf, but we must include <stdio.h>.
- The host code will create an instance of `MyStruct`, initialize it, launch the kernel with a single thread, and then synchronize.
- We'll include basic error checking after the kernel launch.
- The entire code is self‑contained in a single .cu file and can be compiled with `nvcc`.

Code:
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct MyStruct {
    float vec[3];
};

__global__ void printStruct(MyStruct s) {
    // Print the array elements from device
    printf("Device: vec[0] = %f, vec[1] = %f, vec[2] = %f\n",
           s.vec[0], s.vec[1], s.vec[2]);
}

int main() {
    // Initialize host struct
    MyStruct h_s = { { 1.0f, 2.0f, 3.0f } };

    // Launch kernel with a single thread
    printStruct<<<1, 1>>>(h_s);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA device sync error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Host prints confirmation
    printf("Host: Kernel finished execution.\n");

    return 0;
}
```