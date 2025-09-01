```c
/*
Aim: Write to a negative index, like `d_ptr[-1]`, and check the report.

Thinking:
- We want to create a minimal CUDA program that deliberately performs an out‑of‑bounds write.
- Allocate a small array on the device (e.g., 10 integers) and launch a kernel that writes to the element just before the start of the array (`d_ptr[-1]`). This is an invalid memory access.
- After launching the kernel we must query CUDA for launch errors with `cudaGetLastError()`. However, errors that occur during kernel execution (like invalid memory accesses) are only reported after a synchronization point, so we also call `cudaDeviceSynchronize()`.
- We will print any error messages returned by these CUDA API calls. This will show the CUDA runtime detecting the out‑of‑bounds write.
- Finally, we clean up the allocated device memory. We also optionally copy back a valid element (e.g., `d_ptr[0]`) to verify that normal accesses still work.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeNegativeIndex(int *d_ptr)
{
    // Intentionally write to a negative index (out-of-bounds)
    d_ptr[-1] = 123;  // This will cause an illegal memory access
    // Also write to a valid index for completeness
    d_ptr[0] = 42;
}

int main(void)
{
    int *d_ptr = NULL;
    const size_t num_elements = 10;
    const size_t size = num_elements * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch kernel with 1 block and 1 thread
    writeNegativeIndex<<<1, 1>>>(d_ptr);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel launched successfully.\n");
    }

    // Synchronize to catch runtime errors such as illegal memory accesses
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel executed without runtime errors.\n");
    }

    // Optionally copy back a valid element to show normal access works
    int h_value = 0;
    err = cudaMemcpy(&h_value, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
        printf("Value at d_ptr[0] after kernel: %d\n", h_value);
    } else {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }

    // Clean up
    cudaFree(d_ptr);
    return 0;
}
```