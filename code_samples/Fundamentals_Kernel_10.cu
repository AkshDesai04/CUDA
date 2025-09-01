```c
/*
Define a simple C struct on the host. Pass an instance of this struct by value to a kernel. The kernel should print one of the struct's members to verify it was received correctly.

Thinking:
- We create a plain C struct named `MyStruct` containing two members: an integer `a` and a float `b`.
- On the host side we instantiate this struct, fill it with sample values, and pass it directly to a CUDA kernel by value.
- The kernel receives the struct as a parameter, accesses its members, and uses `printf` to output one of them. Device printf requires an architecture of at least sm_30, so we assume the compilation target supports this.
- After launching the kernel with a single thread, we synchronize to ensure the print completes before the program exits.
- Error checking is omitted for brevity but could be added in a production setting.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct MyStruct {
    int a;
    float b;
};

__global__ void printStruct(MyStruct s) {
    printf("Value of a: %d\n", s.a);
}

int main() {
    MyStruct h_s;
    h_s.a = 42;
    h_s.b = 3.14f;

    // Launch kernel with 1 block and 1 thread
    printStruct<<<1, 1>>>(h_s);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    return 0;
}
```