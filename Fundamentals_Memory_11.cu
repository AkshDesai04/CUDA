```cuda
/*
Aim of the program: 
Define a __constant__ memory variable in your CUDA file (e.g., __constant__ int my_const;). 
In the host code, use cudaMemcpyToSymbol to copy a value (e.g., 42) into this variable. 
Write a kernel that reads from my_const and writes its value into an output array.

Thinking process:
1. Declare a constant memory variable: `__constant__ int my_const;`
2. In the host, set a host integer variable `int h_value = 42;`.
3. Use `cudaMemcpyToSymbol(my_const, &h_value, sizeof(int));` to copy the value to device constant memory.
4. Allocate a device output array of ints with `cudaMalloc`.
5. Write a simple kernel that copies the constant value into each element of the output array.
   The kernel will launch a 1D grid where each thread writes the constant to its corresponding index.
6. Allocate a host output array, copy the device array back with `cudaMemcpy`.
7. Print the first few elements to verify that each element equals 42.
8. Add basic error checking after CUDA API calls for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Constant memory variable
__constant__ int my_const;

// Kernel that writes the constant value into an output array
__global__ void write_const_to_array(int *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = my_const;
    }
}

// Helper macro for error checking
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void)
{
    const int N = 256;           // Size of output array
    const int threadsPerBlock = 64;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host value to copy to constant memory
    int h_value = 42;
    CUDA_CHECK(cudaMemcpyToSymbol(my_const, &h_value, sizeof(int)));

    // Allocate device memory for output array
    int *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_output, N * sizeof(int)));

    // Launch kernel
    write_const_to_array<<<blocks, threadsPerBlock>>>(d_output, N);
    CUDA_CHECK(cudaGetLastError());

    // Allocate host memory to retrieve results
    int h_output[N];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify and print first 10 elements
    printf("First 10 elements of the output array:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```