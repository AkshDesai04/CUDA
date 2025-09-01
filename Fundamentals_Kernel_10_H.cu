/*
Create a `__device__` function that takes the struct as an argument by value.
Thinking:
- Define a simple struct (MyStruct) with a couple of fields.
- Implement a __device__ function that accepts MyStruct by value and performs a trivial operation.
- Provide a kernel that calls this device function.
- Include minimal host code to launch the kernel for demonstration.
- The focus is on illustrating passing a struct by value to a __device__ function.
*/

#include <cuda_runtime.h>
#include <stdio.h>

struct MyStruct {
    int   id;
    float value;
};

// __device__ function that takes MyStruct by value
__device__ void processStruct(MyStruct s) {
    // Simple operation: increment id and scale value
    s.id += 1;
    s.value *= 2.0f;
    // In a real scenario, we might write results back to global memory,
    // but for this example we just demonstrate the pass-by-value semantics.
}

// Kernel that calls the __device__ function
__global__ void kernelProcess(MyStruct s) {
    // Each thread processes its own copy
    processStruct(s);
    // For demonstration, print from device (requires compute capability >= 2.0)
    printf("Thread %d processed struct: id=%d, value=%f\n", threadIdx.x, s.id, s.value);
}

int main() {
    // Prepare struct on host
    MyStruct h_struct = {42, 3.14f};

    // Allocate device memory for struct (to pass by value via kernel argument)
    MyStruct *d_struct;
    cudaMalloc((void**)&d_struct, sizeof(MyStruct));
    cudaMemcpy(d_struct, &h_struct, sizeof(MyStruct), cudaMemcpyHostToDevice);

    // Launch kernel with 4 threads for demonstration
    kernelProcess<<<1, 4>>>(*d_struct);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_struct);

    return 0;
}
