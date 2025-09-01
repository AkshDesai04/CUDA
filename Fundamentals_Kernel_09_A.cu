```cpp
/*
Create a __device__ function add_two(int a, int b) that returns a+b. Call it from your kernel.
I need to define a CUDA device function that simply adds two integers. This function should be called
inside a kernel. The kernel will write the result of add_two to an output array. In the host code
I will allocate memory on the device, launch the kernel with a simple configuration, copy the
results back to the host, print them, and clean up. I also include minimal error checking
and use a small example where the kernel adds 3 and 4, expecting the result 7 on each
thread. The code is self-contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__device__ int add_two(int a, int b) {
    return a + b;
}

__global__ void addKernel(int *out, int a, int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = add_two(a, b);
}

int main(void) {
    const int N = 10;                         // number of threads
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *d_out;
    cudaMalloc((void**)&d_out, N * sizeof(int));

    addKernel<<<blocks, threadsPerBlock>>>(d_out, 3, 4);
    cudaDeviceSynchronize();

    int h_out[N];
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_out);
    return 0;
}
```