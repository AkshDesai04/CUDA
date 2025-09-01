/*
Use a `float*` instead of an `int*` and write the value `3.14f`.

Thinking:
The user asks for a CUDA program that uses a `float*` rather than an `int*` and writes the value `3.14f`. To satisfy this, we will create a minimal CUDA C program that:
1. Allocates a single float on the device.
2. Launches a kernel that writes the constant `3.14f` into that device memory.
3. Copies the result back to host memory.
4. Prints the value to confirm it was written correctly.

We will use the standard CUDA runtime API. Error checking will be performed with a simple macro to keep the code concise. The program will compile with `nvcc` and run on any CUDA-capable device.

The structure of the code:
- Include necessary headers (`stdio.h`, `cuda_runtime.h`).
- Define a macro `CUDA_CHECK` for error handling.
- Write a simple kernel `set_value` that dereferences a `float*` and assigns `3.14f`.
- In `main`, allocate device memory (`cudaMalloc`), launch the kernel, copy back (`cudaMemcpy`), print the result, and free resources.

This fulfills the requirement of using a `float*` and writing `3.14f`. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

__global__ void set_value(float *d_val)
{
    *d_val = 3.14f;
}

int main(void)
{
    float *d_val = NULL;
    float h_val = 0.0f;

    // Allocate device memory for a single float
    CUDA_CHECK(cudaMalloc((void**)&d_val, sizeof(float)));

    // Launch kernel with one thread to set the value
    set_value<<<1, 1>>>(d_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_val, d_val, sizeof(float), cudaMemcpyDeviceToHost));

    // Print the value
    printf("Value on host after kernel execution: %f\n", h_val);

    // Clean up
    CUDA_CHECK(cudaFree(d_val));

    return 0;
}
