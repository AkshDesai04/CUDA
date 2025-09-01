```cpp
/*
Define a nested struct (a struct that contains another struct as a member). Pass it by value to a kernel and access a member of the inner struct.

Thinking:
- Create a simple nested struct: `Inner` inside `Outer`.
- Ensure both structs are POD (plain old data) so they can be passed by value to a kernel.
- Write a kernel that receives an `Outer` instance by value and accesses `o.inner.a` and `o.inner.b`.
- Use `printf` in the kernel to output the values.
- In `main`, populate an `Outer` object, launch the kernel with one thread, and synchronize.
- Include necessary headers (`stdio.h`, `cuda_runtime.h`) and compile with nvcc.
- No additional files or text; only the complete .cu file is provided.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct Inner {
    int   a;
    float b;
};

struct Outer {
    Inner inner;
    int   c;
};

__global__ void kernel(Outer o) {
    // Access the inner struct's members and print them
    printf("Inner a: %d, b: %f, Outer c: %d\n", o.inner.a, o.inner.b, o.c);
}

int main(void) {
    // Create and initialize an Outer object on the host
    Outer h_outer;
    h_outer.inner.a = 42;
    h_outer.inner.b = 3.14f;
    h_outer.c       = 7;

    // Launch kernel with one thread; the struct is passed by value
    kernel<<<1, 1>>>(h_outer);

    // Wait for kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```