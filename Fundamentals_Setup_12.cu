/*
Create a CUDA program that deliberately fails to compile due to a syntax error within the `__global__` function (e.g., missing a semicolon). Compile it with `nvcc` and carefully analyze the error message produced. This helps in learning how to debug compilation issues.

Thinking:
I need to write a simple CUDA program that compiles under nvcc except for a deliberate syntax error inside a __global__ kernel. A missing semicolon after an assignment line is a classic, easy-to‑spot mistake. The kernel will be called from a host main function that allocates memory, launches the kernel, copies data back, and prints it. Since the kernel contains a syntax error, nvcc will emit an error message pointing to the problematic line. The code is kept minimal to focus on the compile error rather than performance or correctness. The program should be saved as a .cu file and compiled with nvcc, e.g., `nvcc -arch=sm_35 test.cu -o test`. The expected error will highlight the missing semicolon.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel(int *d_arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x
    d_arr[idx] = idx; // missing semicolon after idx assignment
}

int main() {
    int N = 10;
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));

    myKernel<<<1, N>>>(d_arr);
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    free(h_arr);
    return 0;
}