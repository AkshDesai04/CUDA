```c
/* 
Define a `__constant__` memory variable in your CUDA file (e.g., `__constant__ int my_const;`). In the host code, use `cudaMemcpyToSymbol` to copy a value (e.g., 42) into this variable. Write a kernel that reads from `my_const` and writes its value into an output array.
The goal of this program is to demonstrate how to set a value in constant memory from the host, then have a kernel read that value and store it into an array on the device. We first declare a `__constant__` variable `my_const` at global scope. In the host, we set an `int` variable `val` to 42 and copy it into constant memory using `cudaMemcpyToSymbol`. A simple kernel `write_const_to_array` writes `my_const` into every element of a device array; we launch enough threads to cover the array length. After kernel execution, we copy the device array back to the host and print its contents to confirm that all elements equal 42. We include basic error checking to catch any CUDA API failures. The program is self‑contained and can be compiled with `nvcc -o const_mem const_mem.cu` and executed to observe the output. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Define a constant memory variable */
__constant__ int my_const;

/* Kernel that writes the constant value into an output array */
__global__ void write_const_to_array(int *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = my_const;
    }
}

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    const int N = 10;              // Size of output array
    int *d_out = NULL;             // Device output array
    int h_out[N];                  // Host output array

    /* Allocate device memory for output array */
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(int)));

    /* Host value to copy into constant memory */
    int val = 42;
    /* Copy value to constant memory */
    CHECK_CUDA(cudaMemcpyToSymbol(my_const, &val, sizeof(int)));

    /* Launch kernel: 1 block with N threads (for simplicity) */
    dim3 blockDim(32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    write_const_to_array<<<gridDim, blockDim>>>(d_out, N);
    CHECK_CUDA(cudaGetLastError());   // Check for kernel launch errors

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Print results */
    printf("Output array contents:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    /* Clean up */
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
```