/*
 * Aim of the program:
 * Define a struct `MyData { int a; float b; }`. In the host, initialize it with `{10, 3.14f}`.
 * Pass it to the kernel and have the kernel print both members.
 *
 * Thinking:
 * 1. Include necessary headers: <cstdio> for printf, <cuda_runtime.h> for CUDA API.
 * 2. Define the struct MyData with the specified fields.
 * 3. Implement a __global__ kernel that takes a MyData argument by value.
 *    The kernel will use device-side printf to output the two members.
 * 4. In main():
 *    - Create an instance of MyData on the host and initialize it.
 *    - Launch the kernel with a single thread (<<<1,1>>>) passing the struct.
 *    - Synchronize and check for errors.
 * 5. Ensure that device printf is available (requires compute capability >= 2.0).
 * 6. Keep the code self-contained; no external files or libraries are required.
 */

#include <cstdio>
#include <cuda_runtime.h>

struct MyData {
    int a;
    float b;
};

__global__ void printMyDataKernel(MyData data) {
    // Device-side printf to output struct members
    printf("Kernel received MyData: a = %d, b = %f\n", data.a, data.b);
}

int main() {
    // Host-side initialization
    MyData h_data = {10, 3.14f};

    // Launch kernel with a single thread
    printMyDataKernel<<<1, 1>>>(h_data);

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
