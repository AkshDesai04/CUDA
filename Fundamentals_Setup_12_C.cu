/*
Aim of the program: Create an error by forgetting the `<<<...>>>` syntax for the kernel launch.

Thinking:
The goal is to write a CUDA program that compiles to a binary, but introduces a deliberate syntax error by omitting the triple-chevron kernel launch syntax. This will cause the compiler to treat the kernel as a regular function call, leading to a compile-time error such as "function call missing '<<<...>>>'". The program will therefore not compile successfully, demonstrating the importance of the kernel launch syntax.

To achieve this, I will:
1. Define a simple CUDA kernel (`add`) that performs element-wise addition of two arrays.
2. In `main`, allocate device memory for input and output arrays.
3. Copy input data from host to device.
4. Call the kernel without the `<<<>>>` syntax: `add(a_d, b_d, c_d, n);` instead of `add<<<grid, block>>>(a_d, b_d, c_d, n);`.
5. The rest of the program will be otherwise correct, so the error will be due to the missing launch syntax.

The code below intentionally omits the launch syntax to create the error as requested. 
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void add(const int *a, const int *b, int *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    const int N = 10;
    size_t size = N * sizeof(int);

    int h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Intentionally omitted <<<grid, block>>> launch syntax to create an error
    add(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Result:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_c[i] << " ";
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
