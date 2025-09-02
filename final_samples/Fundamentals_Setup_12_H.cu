```cuda
/*
Aim: Deliberately mismatch the type of an argument, e.g., pass a `float` where the kernel expects an `int*`.

Thinking:
- The goal is to produce a compile‑time error by calling a CUDA kernel with an argument that does not match its declared type.
- Define a simple kernel that expects a pointer to an integer array (`int *`).
- In the host code, declare a floating point variable (`float h_float`).
- Invoke the kernel with `h_float` instead of an `int*`. This mismatch should trigger a compiler error such as “no matching function for call to ‘kernel(float)’”.
- No attempt is made to cast the float to a pointer; the mismatch is intentional.
- The rest of the code (memory allocation, error checking) is minimal because the focus is on the type mismatch.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(int *d_arr) {
    // Dummy kernel: just store the thread index into the array
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_arr[idx] = idx;
}

int main(void) {
    float h_float = 3.14f;           // Host float variable

    // Intentionally invoke the kernel with a float argument instead of int*
    // This will cause a compile-time error.
    kernel<<<1, 1>>>(h_float);       // <-- mismatched type

    cudaDeviceSynchronize();
    return 0;
}
```