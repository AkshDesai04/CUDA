```cpp
/*
Aim of the program:
Define a struct that contains a small, fixed-size array (e.g., `float vec[3]`). Pass it by value and have the kernel print the array elements.

Thinking process:
- To demonstrate passing a small struct by value to a CUDA kernel, I will define a struct named `SmallStruct` that contains a fixed-size array `float vec[3]`.
- The kernel `printStruct` will accept an instance of `SmallStruct` as its argument, which ensures the struct is passed by value. Inside the kernel, I will use `printf` to output each element of the array. CUDA supports `printf` on device code when the compute capability is at least 2.0.
- On the host side, I will create an instance of `SmallStruct`, initialize the array elements, and launch the kernel with a single thread and a single block to keep the example minimal.
- I will include basic error checking after kernel launch and after `cudaDeviceSynchronize()` to catch any CUDA errors.
- Finally, I will call `cudaDeviceReset()` before exiting to clean up the device.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Struct with a small fixed-size array
struct SmallStruct {
    float vec[3];
};

// Kernel that receives the struct by value and prints its elements
__global__ void printStruct(SmallStruct s) {
    // Print from the first (and only) thread
    printf("Thread %d: vec[0] = %f, vec[1] = %f, vec[2] = %f\n",
           threadIdx.x, s.vec[0], s.vec[1], s.vec[2]);
}

int main() {
    // Create and initialize the struct on the host
    SmallStruct hostStruct = { {1.1f, 2.2f, 3.3f} };

    // Launch kernel with 1 block of 1 thread
    printStruct<<<1, 1>>>(hostStruct);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel execution: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Clean up
    cudaDeviceReset();

    return 0;
}
```