/*
Define a struct `MyData { int a; float b; }`. In the host, initialize it with `{10, 3.14f}`. Pass it to the kernel and have the kernel print both members.

Thinking:
- Need a struct definition.
- Host initialization.
- Kernel that accepts MyData and prints members.
- Device printf requires GPU architecture sm_30+.
- Kernel launch with one block one thread.
- Optionally check for errors.
- Use CUDA runtime API.
- Use printf from device, so include <cstdio>.
- After kernel launch, synchronize and exit.
*/

#include <cstdio>
#include <cuda_runtime.h>

struct MyData {
    int a;
    float b;
};

__global__ void printData(MyData d) {
    printf("MyData: a = %d, b = %f\n", d.a, d.b);
}

int main() {
    // Host initialization of MyData
    MyData h_data = {10, 3.14f};

    // Launch kernel with a single thread
    printData<<<1, 1>>>(h_data);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
