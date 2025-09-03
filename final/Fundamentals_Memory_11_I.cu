```cuda
/*
Try to use `cudaMemcpyFromSymbol` to read the value of the constant variable back to the host.

My thinking:  
I need a minimal CUDA program that demonstrates how to copy data into a device constant memory
using `cudaMemcpyToSymbol` and then copy it back to the host using `cudaMemcpyFromSymbol`.  
A constant variable must be declared with `__constant__`.  
To change its value I use `cudaMemcpyToSymbol` (since it is read‑only from kernels).  
After that I use `cudaMemcpyFromSymbol` to read it back.  
Error checking helper will make debugging easier.  
The program prints the value read back, confirming that the round‑trip worked.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Device constant memory variable
__constant__ int d_const;

// Kernel that reads the constant variable (optional, just for demonstration)
__global__ void kernel_read_const(int *out) {
    *out = d_const;
}

int main(void) {
    // Host value to copy into constant memory
    int h_val = 42;

    // Copy host value to device constant memory
    CHECK(cudaMemcpyToSymbol(d_const, &h_val, sizeof(int), 0, cudaMemcpyHostToDevice));

    // Allocate device memory for kernel output (optional)
    int *d_out;
    CHECK(cudaMalloc(&d_out, sizeof(int)));

    // Launch kernel that reads the constant variable
    kernel_read_const<<<1,1>>>(d_out);
    CHECK(cudaDeviceSynchronize());

    // Copy result from kernel back to host
    int h_out = 0;
    CHECK(cudaMemcpyFromSymbol(&h_out, d_const, sizeof(int), 0, cudaMemcpyDeviceToHost));

    printf("Value copied to constant memory: %d\n", h_val);
    printf("Value read back via cudaMemcpyFromSymbol: %d\n", h_out);

    // Optional: also read kernel output
    int kernel_out = 0;
    CHECK(cudaMemcpy(&kernel_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Kernel read constant value: %d\n", kernel_out);

    // Clean up
    CHECK(cudaFree(d_out));

    return 0;
}
```