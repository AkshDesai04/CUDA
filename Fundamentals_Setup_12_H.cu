/*
Aim of the program:
Deliberately mismatch the type of an argument, e.g., pass a `float` where the kernel expects an `int*`.

Thinking behind the code:
1. The kernel is defined to accept an `int*` parameter and writes thread indices into it.
2. In `main`, a `float` variable is declared.
3. The kernel is invoked with this `float` variable as the argument, which is a type mismatch.
4. This mismatch will cause a compile‑time error in CUDA (no known conversion from `float` to `int*`).
5. The program is intentionally written to demonstrate this error; it is not meant to run successfully.
6. To observe the error, compile with: `nvcc mismatch_type.cu -o mismatch_type`.
7. The compiler will report something like: "no matching function for call to `kernel`" and indicate the type mismatch.
*/

#include <stdio.h>

__global__ void kernel(int *d_array) {
    int idx = threadIdx.x;
    d_array[idx] = idx;
}

int main() {
    float f = 3.14f; // A float variable

    // Intentional type mismatch: pass a float where an int* is expected
    kernel<<<1, 1>>>(f); // This line will cause a compilation error

    cudaDeviceSynchronize();
    return 0;
}
