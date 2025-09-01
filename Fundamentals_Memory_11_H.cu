```cpp
/*
Aim of the program:
Use the alternative syntax for `cudaMemcpyToSymbol` which takes the symbol as a variable: `cudaMemcpyToSymbol(my_const, ...)`.

Thinking:
The goal is to demonstrate how to copy data from host memory to a device constant memory symbol using the alternative
syntax where the symbol is passed as a variable rather than as a string literal. We define a constant array `my_const`
in device memory, copy a host array to it, launch a simple kernel that uses the constant array, and then copy the
results back to the host for verification. The code includes basic error checking and uses `cudaMemcpyToSymbol`
with the symbol variable form: `cudaMemcpyToSymbol(my_const, ...)`. It also showcases that the constant memory can be
accessed efficiently within a kernel. The program is fully self‑contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 5

// Device constant memory symbol
__constant__ int my_const[N];

// Kernel that uses the constant memory symbol
__global__ void doubleConst(int *out)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        // Each thread reads its corresponding element from constant memory
        out[idx] = my_const[idx] * 2;
    }
}

// Utility macro for error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    // Host data
    int h_in[N]  = {1, 2, 3, 4, 5};
    int h_out[N] = {0};

    // Allocate device output array
    int *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // Copy host data to device constant memory using the alternative syntax
    // Note: the symbol is passed as a variable (my_const) rather than as a string.
    CUDA_CHECK(cudaMemcpyToSymbol(my_const, h_in, N * sizeof(int)));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleConst<<<blocks, threadsPerBlock>>>(d_out);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Input array: ");
    for (int i = 0; i < N; ++i)
        printf("%d ", h_in[i]);
    printf("\n");

    printf("Output array (double of constant): ");
    for (int i = 0; i < N; ++i)
        printf("%d ", h_out[i]);
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
```