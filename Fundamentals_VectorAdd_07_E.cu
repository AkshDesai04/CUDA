```c
/*
Aim of the program:
Implement a kernel that computes `x^n` where `n` is passed as a parameter: `output[i] = powf(input[i], n)`.

Thinking:
The task is to write a complete CUDA C program that defines a kernel performing a power operation on an array of floating‑point numbers.
1. The kernel will read from an input array, raise each element to the power `n`, and write the result to an output array.
2. `n` will be passed to the kernel as a float because the CUDA intrinsic `powf` takes a float exponent.
3. The program will:
   - Allocate host memory for a small sample array, e.g. 10 elements.
   - Initialize the array with some values.
   - Allocate device memory for input and output arrays.
   - Copy the input data to the device.
   - Launch the kernel with a grid of enough blocks to cover all elements.
   - Copy the result back to the host.
   - Print both input and output to verify correctness.
4. Error checking will be performed after each CUDA API call using a helper macro `CUDA_CHECK`.
5. The kernel uses the common pattern `int idx = blockIdx.x * blockDim.x + threadIdx.x;` and guards against out‑of‑bounds indices.
6. The grid and block dimensions are chosen such that each thread processes at most one element; for a small array a single block is sufficient, but the code works for larger arrays as well.
7. The program is self‑contained and can be compiled with `nvcc -o power power.cu`.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void powerKernel(const float *input, float *output, float exponent, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = powf(input[idx], exponent);
    }
}

int main(void)
{
    const int N = 10;
    float h_input[N];
    float h_output[N];

    // Initialize input array with some values
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i + 1); // 1.0, 2.0, ..., 10.0
    }

    float *d_input = NULL;
    float *d_output = NULL;
    size_t bytes = N * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_output, bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Set exponent value
    float exponent = 3.0f; // Example: cube each element

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    powerKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, exponent, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input\tOutput (power %f)\n", exponent);
    for (int i = 0; i < N; ++i) {
        printf("%f\t%f\n", h_input[i], h_output[i]);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```