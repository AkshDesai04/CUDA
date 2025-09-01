```cpp
/*
Pass the struct pointer, but forget to copy the data from host to device first.
What values does the kernel see when it reads from the struct's members?

Thinking:
In CUDA, when you allocate memory on the device with cudaMalloc, that memory is
uninitialized. It contains whatever bytes happened to be in that region of device
memory previously, which is effectively garbage. If you pass a pointer to such
memory to a kernel and the kernel reads the struct members, it will see those
garbage values, not the values you set on the host. Therefore the kernel will
print arbitrary numbers for each member, typically random integers/floats and
a non‑null-terminated string (or any leftover data). This demonstrates the
importance of copying data from host to device before using it in a kernel.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct MyStruct {
    int   a;
    float b;
    char  c[10];
};

__global__ void readStruct(MyStruct *s) {
    // Use printf inside kernel to print the struct members
    printf("Kernel sees: a = %d, b = %f, c = \"%s\"\n", s->a, s->b, s->c);
}

int main() {
    // Host side struct initialization
    MyStruct h_struct;
    h_struct.a = 42;
    h_struct.b = 3.14f;
    snprintf(h_struct.c, sizeof(h_struct.c), "Hello");

    // Allocate device memory for the struct
    MyStruct *d_struct;
    cudaError_t err = cudaMalloc((void**)&d_struct, sizeof(MyStruct));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Intentionally do NOT copy h_struct to d_struct
    // cudaMemcpy(d_struct, &h_struct, sizeof(MyStruct), cudaMemcpyHostToDevice);

    // Launch kernel
    readStruct<<<1,1>>>(d_struct);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_struct);
    cudaDeviceReset();

    return 0;
}
```