/*
Aim: Create a syntax error by misspelling `__global__` as `__globol__` and observe the compiler error.

Thinking:
The goal is to produce a compilation error that highlights how a misspelled CUDA kernel declaration keyword results in a syntax error. The code will define a simple CUDA kernel that writes indices into an array, launch it from the host, copy back the results, and print them. By intentionally misspelling `__global__` as `__globol__`, the compiler will not recognize the keyword, leading to a clear error message that can be observed during compilation. The rest of the program is a standard minimal CUDA example to demonstrate a functional context for the error.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__globol__ void myKernel(int *d_arr) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    d_arr[idx] = idx;
}

int main() {
    const int N = 10;
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));

    myKernel<<<1, N>>>(d_arr);

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
