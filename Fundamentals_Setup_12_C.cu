/*
Aim: Create an error by forgetting the `<<<...>>>` syntax for the kernel launch.

Thinking: I will write a simple kernel that adds two arrays. Then I will launch the kernel incorrectly without the triple angle brackets, causing a compile error. The code will use cudaMalloc, cudaMemcpy, and free, but the kernel call will be wrong. This demonstrates the importance of the launch syntax.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add_arrays(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    const int N = 10;
    const size_t size = N * sizeof(float);
    float h_a[N], h_b[N], h_c[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Incorrect kernel launch: missing <<<...>>> syntax
    add_arrays(d_a, d_b, d_c, N);  // This line will cause a compilation error

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
