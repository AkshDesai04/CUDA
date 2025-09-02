/*
Aim: Modify the kernel to change a member of the struct it received. Does this change affect the original struct on the host? Explain why or why not (pass-by-value).

Thinking:
In CUDA, kernel arguments are passed by value. When we pass a struct to a kernel, the struct is copied to the device memory for the duration of the kernel call. Any modifications to that struct inside the kernel affect only the copy that exists on the device; the original struct on the host remains unchanged unless we explicitly copy data back from device to host. In this example, we will define a simple struct, pass it to a kernel that modifies its fields, and then observe that the host struct's fields are unchanged after the kernel execution, demonstrating pass‑by‑value semantics.
*/

#include <stdio.h>
#include <cuda.h>

struct MyStruct {
    int a;
    float b;
};

__global__ void modifyStruct(MyStruct s) {
    // Modify the struct fields (this is a copy on the device)
    s.a = 42;
    s.b = 3.14f;
    printf("Inside kernel: a = %d, b = %f\n", s.a, s.b);
}

int main() {
    // Initialize the struct on the host
    MyStruct h_s;
    h_s.a = 10;
    h_s.b = 2.718f;

    printf("Before kernel: a = %d, b = %f\n", h_s.a, h_s.b);

    // Launch kernel with one thread
    modifyStruct<<<1,1>>>(h_s);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // After kernel execution, the host struct remains unchanged
    printf("After kernel: a = %d, b = %f\n", h_s.a, h_s.b);

    return 0;
}