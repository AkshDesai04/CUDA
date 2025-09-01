/*
Aim of the program: 
Write a kernel that takes two pointers, int* in and int* out. It should read the value from in, multiply it by 2, and write the result to out.

Thinking process:
1. The kernel needs to perform a simple arithmetic operation on a single integer. Since there's only one input and one output value, a single thread is sufficient. 
2. The kernel signature will be `__global__ void multiply_by_two(int* in, int* out)`. Inside, it simply does `*out = 2 * (*in);`.
3. The host code will:
   - Allocate host memory for an integer input and an integer output.
   - Allocate device memory for the input and output.
   - Copy the input from host to device.
   - Launch the kernel with one block and one thread.
   - Copy the result back from device to host.
   - Print the result to verify correctness.
4. Use simple error checking for CUDA API calls.
5. Include necessary headers and define the `CHECK_CUDA` macro for concise error handling.
6. The program will compile to a .cu file and can be built with `nvcc -o multiply multiply.cu`.

The code below implements this plan.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Kernel that reads an integer from 'in', multiplies it by 2, and writes to 'out'.
__global__ void multiply_by_two(int* in, int* out) {
    // Since we have only one value, we don't need thread indexing.
    *out = 2 * (*in);
}

int main(void) {
    int h_in = 7;           // Example input value
    int h_out = 0;          // Placeholder for output

    int *d_in = NULL;
    int *d_out = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_in, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(int)));

    // Copy input from host to device
    CHECK_CUDA(cudaMemcpy(d_in, &h_in, sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with 1 block of 1 thread
    multiply_by_two<<<1, 1>>>(d_in, d_out);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Print result
    printf("Input: %d, Output (multiplied by 2): %d\n", h_in, h_out);

    // Free device memory
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
