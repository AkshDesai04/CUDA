/*
Define a struct `MyData { int a; float b; }`. In the host, initialize it with `{10, 3.14f}`. Pass it to the kernel and have the kernel print both members.

[Thinking: Need to create a small CUDA program that demonstrates passing a struct from host to device. We'll define the struct. Host will create an instance, initialize with given values. We'll launch a kernel that takes the struct as an argument by value. Inside the kernel, use device printf to output the struct members. Then sync and exit. We need to include stdio.h for printf, and the CUDA runtime header. We also add a small error checking macro. The kernel can be simple. We'll use a single thread grid. We'll call cudaDeviceSynchronize to wait for kernel. The device printf will output on the console. We'll compile with nvcc. We'll also add return 0; at end. No extra outputs. That's all. 
*/
#include <stdio.h>
#include <cuda_runtime.h>

struct MyData {
    int a;
    float b;
};

__global__ void printMyData(MyData d) {
    printf("a = %d, b = %f\n", d.a, d.b);
}

int main() {
    MyData h_data = {10, 3.14f};

    // Launch kernel with one thread
    printMyData<<<1, 1>>>(h_data);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    return 0;
}
